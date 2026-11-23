# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MoonEP 对 FSDP 参数恢复和梯度归约前扩展接口的适配。"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import bind_execution


class MoonEPAllGatherExtension:
    """Continue replica preparation from FSDP-owned parameter restoration hooks."""

    def __init__(self, executor: Any, runtime: Any) -> None:
        """Bind one expert unit to its shared weight runtime."""
        self.executor = executor
        self.runtime = runtime
        self.signature = inspect.signature(executor.owner.forward)

    def fsdp_pre_forward(self, _state: Any, args: tuple, kwargs: dict) -> None:
        """Capture FSDP's cast routing inputs before restoring expert parameters."""
        values = args[:3] if len(args) >= 3 else tuple(self.signature.bind(*args, **kwargs).arguments.values())
        if len(values) != 3 or self.executor.layer in self.runtime.forward_inputs:
            raise RuntimeError("MoonEP requires one pending (hidden, ids, scores) invocation per expert module")
        key = bind_execution(self.executor, lambda: self.runtime.new_key(self.executor))
        self.runtime.forward_inputs[self.executor.layer] = key, values

    def fsdp_pre_all_gather(self, _state: Any) -> None:
        """Observe FSDP restoring its own parameter storage."""

    def fsdp_post_unshard(self, _state: Any) -> None:
        """Prepare replicas after home parameters become ready."""
        self._prefetch()

    def _prefetch(self) -> None:
        """Select a retained invocation and submit its backend's replica preparation."""
        pending = self.runtime.forward_inputs.get(self.executor.layer)
        if pending is not None:
            key, values = pending
            if key in self.runtime.weight_leases:
                return
            self.executor.prepare_forward(key, *values)
        else:
            key = self.runtime.prefetch_key(self.executor)
        if (key is None or key in self.runtime.weight_leases
                or key in self.runtime.pending_weight_preparations):
            return
        self.runtime.prepare_restored_weights(key)
        statistic = "fsdp_ep_demand_preparations" if pending is not None else "fsdp_ep_prefetches"
        self.runtime.statistics[statistic] += 1


def register_fsdp_extensions(executor: Any, runtime: Any, gradient_type: type | None = None) -> Any:
    """Bind every backend to the same FSDP restoration and gradient hook protocol.

    Args:
        executor: Expert owner whose FSDP wrapping has already been completed.
        runtime: Backend providing replica preparation and gradient ownership.
        gradient_type: Optional adapter for contributions that arrive after custom backward returns.

    Returns:
        The gradient adapter, or None when no deferred contribution needs merging.
    """
    scheduler = getattr(executor.owner, "hsdp_scheduler", None)
    if scheduler is None:
        return None
    if len(scheduler.modules) != 1 or scheduler.cell is not executor.owner:
        raise ValueError("MoonEP experts must form an independent FSDP unit")
    scheduler.enable_param_grad_gate()
    scheduler.register_all_gather_extension(MoonEPAllGatherExtension(executor, runtime))
    runtime.fsdp_layers.add(executor.layer)
    if gradient_type is None:
        return None
    gradient = gradient_type(executor, runtime)
    scheduler.register_pre_reduce_hook(gradient.complete)
    return gradient


class MoonEPGradientExtension:
    """Merge replica gradients before FSDP reduces the real owner buffers.

    Ordinary gradients pass through autograd. With accumulation fusion, home
    products write FP32 g directly and notify the parameter gate explicitly.
    Both paths join pending replica readers before the native reduction.
    """

    def __init__(self, executor: Any, runtime: Any) -> None:
        """绑定一个专家 FSDP 单元；例如 B7 的多次微批共用同一个待归并队列。"""
        self.executor = executor
        self.runtime = runtime
        self.pending = []
        self._parameter_layout = None

    @staticmethod
    def _local(value: torch.Tensor) -> torch.Tensor:
        """取 DTensor 的本地张量；例如 EP 专家分片的梯度变为本卡 [P,I,H]。"""
        return value.to_local() if isinstance(value, DTensor) else value

    def _layouts(self, state: Any) -> tuple[tuple[Any, bool], tuple[Any, bool]]:
        """按真实共享存储绑定两组参数和 GMM 视图；例如 [P,2I,H] 需要转置 dW。"""
        if self._parameter_layout is not None:
            return self._parameter_layout
        parameters = state.hsdp_params
        layouts = []
        for view in self.executor.read_weights():
            view = self._local(view)
            matches = []
            for parameter in parameters:
                local = self._local(parameter.unsharded_param)
                if (view.untyped_storage().data_ptr() != local.untyped_storage().data_ptr()
                        or view.storage_offset() != local.storage_offset()):
                    continue
                if view.shape == local.shape and view.stride() == local.stride():
                    matches.append((parameter, False))
                elif (local.ndim == 3 and view.shape == local.transpose(1, 2).shape
                      and view.stride() == local.transpose(1, 2).stride()):
                    matches.append((parameter, True))
            if len(matches) != 1:
                raise RuntimeError("专家 GMM 视图必须唯一对应真实 FSDP 参数或其二维转置")
            layouts.append(matches[0])
        if len(layouts) != 2 or layouts[0][0] is layouts[1][0]:
            raise RuntimeError("专家梯度扩展要求 gate/up 和 down 分别对应独立参数")
        self._parameter_layout = tuple(layouts)
        return self._parameter_layout

    def uses_accumulated_grad(self, state: Any) -> bool:
        """Honor explicit executor policy or the model's fused-wgrad installation.

        The owner marker is attached by VeOmni only when its gradient accumulation
        fusion switch is enabled.
        """
        enabled = getattr(self.executor, "gradient_accumulation_fusion", None)
        if enabled is False:
            return False
        marked = all(getattr(parameter.unsharded_param, "_veomni_fused_wgrad_owner", None) is parameter
                     for parameter, _ in self._layouts(state))
        if enabled is True and not marked:
            if any(parameter.reduce_comm_dtype() != torch.float32 for parameter, _ in self._layouts(state)):
                raise ValueError("MoonEP fused wgrad requires FP32 FSDP reduction")
            return True
        return marked

    @torch.no_grad()
    def accumulation_buffers(self, state: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Borrow real FSDP FP32 accumulators without resetting earlier microsteps."""
        buffers = []
        for parameter, _ in self._layouts(state):
            if parameter.reduce_comm_dtype() != torch.float32:
                raise ValueError("MoonEP fused wgrad requires FP32 FSDP reduction")
            local = self._local(parameter.unsharded_param)
            if parameter.unsharded_accumulated_grad is None:
                parameter.unsharded_accumulated_grad = torch.zeros_like(local, dtype=torch.float32)
            target = parameter.unsharded_accumulated_grad_data
            if target.dtype != torch.float32 or target.shape != local.shape or not target.is_contiguous():
                raise ValueError("MoonEP accumulation requires contiguous FP32 parameter gradient buffers")
            buffers.append(target)
        return tuple(buffers)

    def notify_accumulated(self, scheduler: Any) -> None:
        """Notify the FSDP readiness gate after side-effect writes are enqueued.

        Returning None to autograd must neither allocate dummy zero gradients nor
        leave the parameter gate waiting for AccumulateGrad hooks that cannot run.
        """
        for parameter, _ in self._layouts(scheduler.hsdp_state):
            scheduler._on_param_grad_ready(parameter.unsharded_param)  # pylint: disable=protected-access

    @torch.no_grad()
    def start(self, gate_up: torch.Tensor, down: torch.Tensor,
              plan: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Submit replica dW and return the local home contribution to autograd.

        Args:
            gate_up: Complete local gate/up dW in GMM layout.
            down: Complete local down dW in GMM layout.
            plan: Frozen forward placement and replica communication metadata.

        Returns:
            The two home contributions in GMM layout, ready for accumulation.
        """
        home_gate_up, home_down, packet = self.runtime.start_gradient_return(self.executor, gate_up, down, plan)
        if packet is not None:
            self.pending.append(packet)
            self.runtime.statistics["gradient_exchanges"] += 1
        return home_gate_up, home_down

    @torch.no_grad()
    def complete(self, state: Any) -> None:
        """等待 EP 并归并到真实参数梯度，再允许原生 RS；例如 B7 的 e0 dW 回 rank 0。

        Args:
            state: 即将执行原生归约的 FSDP 参数状态。
        """
        if not self.pending:
            return
        layouts = self._layouts(state)
        accumulated = self.uses_accumulated_grad(state)
        for packet in self.pending:
            remote_gate_up, remote_down = self.runtime.wait_gradient_return(packet)
            if packet.receive_buffer.device.type != "cpu":
                stream = torch.get_device_module(packet.receive_buffer.device).current_stream()
                packet.plan.send_ids.record_stream(stream)
            for (parameter, transposed), contribution in zip(layouts, (remote_gate_up, remote_down)):
                if accumulated:
                    target = parameter.unsharded_accumulated_grad_data
                    if target is None:
                        raise RuntimeError("MoonEP accumulated gradient is missing before replica merge")
                    if transposed:
                        contribution = contribution.transpose(1, 2)
                    if (len(packet.plan.send_slots) != contribution.shape[0]
                            or any(slot < 0 or slot >= target.shape[0] for slot in packet.plan.send_slots)):
                        raise RuntimeError("replica gradient return has an invalid host owner mapping")
                    for index, slot in enumerate(packet.plan.send_slots):
                        target[slot].add_(contribution[index])
                else:
                    if parameter.unsharded_param.grad is None:
                        raise RuntimeError("FSDP parameter gradient is missing before replica merge")
                    target = self._local(parameter.unsharded_param.grad)
                    if transposed:
                        contribution = contribution.transpose(1, 2).contiguous()
                    target.index_add_(0, packet.plan.send_ids, contribution)
            self.runtime.statistics["gradient_waits"] += 1
        self.pending.clear()
