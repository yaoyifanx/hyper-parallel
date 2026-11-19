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
"""跨层共享权重槽、原生 EFS AG→EP 权重交换及显式专家反向。"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, replace
from numbers import Integral
import os
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import is_replaying
from hyper_parallel.distributed.expert_parallel.moonep.device_planning import TableGraph, select_device_allocation
from hyper_parallel.distributed.expert_parallel.moonep.fsdp_hooks import (
    MoonEPGradientExtension, register_fsdp_extensions,
)
from hyper_parallel.distributed.expert_parallel.moonep.lifecycle import ExecutionKey, MoonEPLifecycle
from hyper_parallel.distributed.expert_parallel.moonep.layout import RouteGeometry, route_geometry
from hyper_parallel.distributed.expert_parallel.moonep.fused.executor import SparseMoonEPExecutor, SparseMoonEPRuntime
from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    ExpertSaved, GroupedWeights, GroupedWeightViews, activation_gradient, expert_backward_values, expert_forward_values,
    grouped_values, grouped_weight_grad, grouped_weight_grad_add,
)
from hyper_parallel.distributed.expert_parallel.moonep.planning import (
    ExecutionPlan, compile_plan, prepare_entry_order,
)
from hyper_parallel.distributed.expert_parallel.moonep.streams import communication_stream
from hyper_parallel.distributed.expert_parallel.moonep.token_dedup import (
    failed_local_dedup, finish_dedup, prepare_local_dedup,
)


def _combine_returned_tokens(hidden: torch.Tensor, returned: torch.Tensor, scores: torch.Tensor,
                             plan: ExecutionPlan, source_rows: torch.Tensor) -> torch.Tensor:
    """Restore original token order while preserving FP32 router-weight gradients."""
    if hidden.device.type == "npu" and returned.numel():
        # torch_npu is optional on non-Ascend hosts, so import only for this native route.
        from hyper_parallel.components.functional.moe_token_unpermute import (  # pylint: disable=C0415
            moe_token_unpermute,
        )

        return moe_token_unpermute(
            returned, plan.unpermute_indices, scores.float().contiguous(),
        ).reshape_as(hidden)
    selected_scores = scores.reshape(-1).index_select(0, plan.dispatch_order)
    weighted = returned * selected_scores[:, None]
    result = torch.zeros_like(hidden.reshape(-1, hidden.shape[-1]), dtype=weighted.dtype)
    return result.index_add(0, source_rows, weighted).to(hidden.dtype).reshape_as(hidden)


@dataclass
class _WeightTableSlot:
    """Final weights and reusable home scratch with independent last-reader fences."""

    gate_up: torch.Tensor
    down: torch.Tensor
    key: ExecutionKey | None = None
    compute_read_done: Any = None
    home_gate_up: torch.Tensor | None = None
    home_down: torch.Tensor | None = None
    home_layout_key: ExecutionKey | None = None
    home_layout_ready: Any = None
    home_layout_read_done: Any = None


@dataclass
class _WeightLease:
    """当前执行键的槽租约与物化完成事件；例如K6被K7预取后直接命中。"""

    slot: _WeightTableSlot
    gate_up: GroupedWeights
    down: GroupedWeights
    ready: Any
    consumed: bool = False


@dataclass
class _GradientPacket:
    """在梯度通信域等待归并的 BF16 副本梯度；例如 B7 的 e0 贡献。"""

    receive_buffer: torch.Tensor
    send_buffer: torch.Tensor
    plan: ExecutionPlan
    gate_up_shape: torch.Size
    down_shape: torch.Size
    work: Any


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """取布局张量的本地可微视图；例如EP的全局[E,H,I]变为本rank[P,H,I]。"""
    return value.to_local() if isinstance(value, DTensor) else value


class MoonEPRuntime(MoonEPLifecycle):
    """整个顺序模型共享按实际并发请求增长的最终 GMM 表。"""

    def __init__(self, executors: list, *, standalone_prefetch: bool = False) -> None:
        """绑定同形状专家单元；例如两个专家请求同时在途时才分配第二槽。"""
        super().__init__(executors)
        # 仅无 FSDP 的独立脚本路径用此开关；FSDP 预取由其专家 AG hook 自然触发。
        self.standalone_prefetch = standalone_prefetch
        self.home_layout_slots: dict[ExecutionKey, _WeightTableSlot] = {}
        self.home_layout_stream = None
        self.compact_weight_slots = os.environ.get("HP_MOONEP_COMPACT_WEIGHTS", "0") == "1"
        self.release_w_workspace = os.environ.get("HP_MOONEP_RELEASE_W_WORKSPACE", "0") == "1"
        self.home_weight_mode = os.environ.get("HP_MOONEP_HOME_WEIGHT_VIEWS", "0")
        if self.home_weight_mode not in ("0", "1", "2"):
            raise ValueError("home weight views must be 0 (stack), 1 (matrix list), or 2 (batched parts)")
        self.home_weight_views = self.home_weight_mode != "0"
        if self.home_weight_views and any(getattr(executor, "chunked", None) is not None for executor in executors):
            raise ValueError("home weight views require the complete nonchunk expert path")
        if self.home_weight_views and any(executor.home_count + executor.replica_capacity > 128
                                         for executor in executors):
            raise ValueError("home weight views require at most 128 admitted local and remote experts")
        self.slots: list[_WeightTableSlot] = []
        self.weight_workspace: dict[str, torch.Tensor] = {}
        self.weight_workspace_demand: dict[str, int] = {}
        self.weight_workspace_busy = False
        self.planning_graphs: dict[tuple, Callable] = {}
        self.stream = None
        self.active_calls = Counter()
        self.shared_executors = set()
        self.capacity = max(executor.home_count + executor.replica_capacity for executor in executors)
        self.weight_send_capacity = max(executor.home_count for executor in executors)
        self.weight_recv_capacity = max(executor.replica_capacity for executor in executors)
        self._validate_planner_capacity()
        self._configure_groups()

    def _validate_planner_capacity(self) -> None:
        """建模时检查各 EP rank 的容量和 chunk 配置一致，避免不同通信 split/轮次。"""
        first = self.executors[0]
        if first.ranks == 1 or not dist.is_initialized():
            return
        local = tuple((executor.experts, executor.replica_capacity, executor.top_k,
                       executor.token_dedup, executor.forward_ratio, executor.backward_ratio,
                       getattr(getattr(executor, "chunked", None), "chunk_rows", 0),
                       self.compact_weight_slots, getattr(executor, "gradient_accumulation_fusion", None),
                       self.home_weight_mode, self.release_w_workspace,
                       os.environ.get("HP_MOONEP_FUSED_WGRAD"), os.environ.get("HP_MOONEP_REPLICA_PREFETCH", "1"))
                      for executor in self.executors)
        gathered = [None] * first.ranks
        dist.all_gather_object(gathered, local, group=first.group)
        if any(other != local for other in gathered):
            raise ValueError("同一 EP 组的 MoonEP 专家、副本、路由、去重和 chunk 配置必须在所有 rank 上一致")

    def _configure_groups(self) -> None:
        """复用 DeviceMesh 的 EP/EFS 域；例如 EP4×EFS2 不再创建相同成员的新组。"""
        for executor in self.executors:
            # Chunk dispatch/return and replica payloads share one ordered group.
            backend = getattr(executor, "chunked", None)
            payload_group = executor.group if backend is None else backend.group
            executor.weight_group = executor.gradient_group = payload_group
            scheduler = getattr(executor.owner, "hsdp_scheduler", None)
            if scheduler is not None:
                state = scheduler.hsdp_state
                groups = {param.mesh_info.shard_process_group
                          for param in state.hsdp_params if param.shard_world_size > 1}
                if len(groups) > 1:
                    raise ValueError("一个专家单元的参数必须使用同一EFS分片组")
                executor.gradient_extension = register_fsdp_extensions(executor, self, MoonEPGradientExtension)

    def new_key(self, executor: Any) -> ExecutionKey:
        """分配原始调用编号；例如同一layer两个微批拥有不同invocation。"""
        key = super().new_key(executor)
        self.active_calls[executor.layer] += 1
        if self.active_calls[executor.layer] > 1:
            self.shared_executors.add(executor.layer)
        return key

    def gradient_stream(self, executor: Any, device: torch.device) -> Any:
        """复用 EP mesh 的唯一通信流；例如 B7 的 dW 接在其权重与 token 通信之后。"""
        if device.type != "npu":
            return None
        if executor.gradient_group is not executor.weight_group or self.stream is None:
            raise RuntimeError("专家梯度必须复用已初始化的 EP mesh 通信流")
        return self.stream

    def start_gradient_return(self, executor: Any, gate_up: torch.Tensor, down: torch.Tensor,
                              plan: ExecutionPlan) -> tuple[torch.Tensor, torch.Tensor, _GradientPacket | None]:
        """在既有 EP 流提交副本回传；例如 B7 的 dW 就绪后让上游反向继续。"""
        stream = self.gradient_stream(executor, gate_up.device) if plan.any_migration else None
        return launch_home_gradients(gate_up, down, plan, executor.gradient_group, stream)

    @staticmethod
    def wait_gradient_return(packet: _GradientPacket) -> tuple[torch.Tensor, torch.Tensor]:
        """在 FSDP 参数梯度就绪后等通信包；例如 B7 的远程 dW 到达再执行 RS。"""
        return wait_home_gradients(packet)

    def _ensure_slots(self, gate_up: torch.Tensor, down: torch.Tensor, capacity: int | None = None) -> None:
        """首次计算只分配一张最终表；例如 BF16 计算不会按 FSDP 深度预留空槽。"""
        if not self.slots:
            if len({id(executor.weight_group) for executor in self.executors}) != 1:
                raise ValueError("共享权重表要求所有专家层共用同一 EP 权重通信域")
            self.stream = (communication_stream(self.executors[0].weight_group, gate_up.device)
                           if gate_up.device.type == "npu" else None)
            self.slots = [self._new_slot(gate_up, down, capacity if self.compact_weight_slots else None)]
        slot = self.slots[0]
        if (slot.gate_up.shape[1:] != gate_up.shape[1:] or slot.down.shape[1:] != down.shape[1:]
                or slot.gate_up.dtype != gate_up.dtype or slot.gate_up.device != gate_up.device):
            raise ValueError("共享专家表槽要求所有专家层的H/I、计算dtype和设备相同")

    def _new_slot(self, gate_up: torch.Tensor, down: torch.Tensor, capacity: int | None = None) -> _WeightTableSlot:
        """按实际在途请求增加一张专家表；例如 B7 占用首槽时 B6 的 hook 获得第二槽。"""
        # 长期表使用既有 EP 通信流的分配池，减少与 C 激活和交叉熵临时量的交错分配。
        capacity = self.capacity if capacity is None else capacity
        with torch.npu.stream(self.stream) if self.stream is not None else nullcontext():
            return _WeightTableSlot(gate_up.new_empty((capacity, *gate_up.shape[1:])),
                                    down.new_empty((capacity, *down.shape[1:])))

    def _acquire_home_slot(self, key: ExecutionKey, gate_up: torch.Tensor, down: torch.Tensor) -> _WeightTableSlot:
        """Reserve a table for layout work or reuse the same pending reservation."""
        capacity = (min(self.capacity, ((len(self.executions[key][1].group_counts) + 3) // 4) * 4)
                    if self.compact_weight_slots else None)
        self._ensure_slots(gate_up, down, capacity)
        if key in self.home_layout_slots:
            return self.home_layout_slots[key]
        slot = next((item for item in self.slots if item.key is None), None)
        if slot is None:
            capacity = (min(self.capacity, ((len(self.executions[key][1].group_counts) + 3) // 4) * 4)
                        if self.compact_weight_slots else None)
            slot = self._new_slot(self.slots[0].gate_up, self.slots[0].down, capacity)
            self.slots.append(slot)
        if self.compact_weight_slots:
            active = len(self.executions[key][1].group_counts)
            if slot.gate_up.shape[0] < active:
                # Retain a small high-water bucket; last readers hold old storage
                # through record_stream, never resize a tensor with queued users.
                capacity = min(self.capacity, ((active + 3) // 4) * 4)
                replacement = self._new_slot(gate_up, down, capacity)
                self.slots[next(index for index, value in enumerate(self.slots) if value is slot)] = replacement
                slot = replacement
                self.statistics["weight_table_growths"] += 1
        slot.key = key
        self.home_layout_slots[key] = slot
        return slot

    @torch.no_grad()
    def prepare_home_layout(self, executor: Any, key: ExecutionKey) -> None:
        """Normalize restored homes without occupying the EP communication stream.

        Args:
            executor: Expert owner whose native FSDP hooks restored its home views.
            key: Current invocation and parameter version, also used by ``prepare``.

        Note:
            Scratch belongs to a final-table slot but is never read by GMM.
            Its next writer waits for W's last scratch reader, independently
            of C's last reader of the final table. All waits stay on device.
        """
        if getattr(self, "home_weight_views", False):
            return
        gate_up, down = executor.read_weights()
        if gate_up.device.type not in ("cpu", "npu") or (gate_up.is_contiguous() and down.is_contiguous()):
            return
        if self.compact_weight_slots and key not in self.executions:
            # The original route compiler has not exported its replica demand.
            # Defer layout until prepare() knows the exact final-table capacity.
            return
        slot = self._acquire_home_slot(key, gate_up, down)
        if slot.home_layout_key == key:
            return
        if gate_up.device.type == 'npu' and self.home_layout_stream is None:
            self.home_layout_stream = torch.npu.Stream(device=gate_up.device)
        if self.home_layout_stream is not None:
            self.home_layout_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.home_layout_stream) if self.home_layout_stream is not None else nullcontext():
            if self.home_layout_stream is not None and slot.home_layout_read_done is not None:
                self.home_layout_stream.wait_event(slot.home_layout_read_done)
            if self.compact_weight_slots:
                if self.home_layout_stream is not None and slot.compute_read_done is not None:
                    self.home_layout_stream.wait_event(slot.compute_read_done)
                slot.home_gate_up = slot.gate_up[:gate_up.shape[0]]
                slot.home_down = slot.down[:down.shape[0]]
            elif slot.home_gate_up is None or slot.home_gate_up.shape != gate_up.shape:
                slot.home_gate_up, slot.home_down = (
                    torch.empty_like(gate_up, memory_format=torch.contiguous_format),
                    torch.empty_like(down, memory_format=torch.contiguous_format),
                )
                self.statistics["home_layout_allocations"] += 1
            if self.home_layout_stream is not None:
                gate_up.record_stream(self.home_layout_stream)
                down.record_stream(self.home_layout_stream)
            slot.home_gate_up.copy_(gate_up)
            slot.home_down.copy_(down)
            if self.home_layout_stream is not None:
                slot.home_layout_ready = torch.npu.Event()
                slot.home_layout_ready.record(self.home_layout_stream)
            slot.home_layout_key = key
        self.statistics["home_layout_preparations"] += 1

    def cancel_home_layout(self, key: ExecutionKey) -> None:
        """Cancel a reservation after planning fails, retaining its stream fences.

        Args:
            key: Invocation whose final table has not yet been consumed.
        """
        slot = self.home_layout_slots.pop(key, None)
        if slot is not None:
            slot.key, slot.home_layout_key = None, None

    @torch.no_grad()
    def prepare(self, key: ExecutionKey) -> _WeightLease:
        """用FSDP已恢复的home视图提交副本交换和组表，消费时才等待ready。"""
        if key.version != self.version:
            raise RuntimeError("旧执行键不能读取更新后的权重")
        if key in self.weight_leases:
            self.statistics["weight_lease_hits"] += 1
            return self.weight_leases[key]
        if self.home_weight_views:
            return self._prepare_weight_views(key)
        executor, plan = self.executions[key]
        # 原生模块hook或预取post已恢复home；本函数只管理副本和计算表。
        if not self.slots:
            gate_up, down = executor.read_weights()
            capacity = (min(self.capacity, ((len(plan.group_counts) + 3) // 4) * 4)
                        if self.compact_weight_slots else None)
            self._ensure_slots(gate_up, down, capacity)
        if self.weight_workspace_busy:
            raise RuntimeError("权重工作区仍在提交或上次物化失败，不能覆盖可能在途的EP输入")
        gate_up, down = executor.read_weights()
        slot = self._acquire_home_slot(key, gate_up, down)
        if not (gate_up.is_contiguous() and down.is_contiguous()) and slot.home_layout_key != key:
            self.prepare_home_layout(executor, key)
        self.home_layout_slots.pop(key, None)
        self.weight_workspace_busy = self.stream is not None
        slot.key = key
        if self.stream is not None:
            self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream) if self.stream is not None else nullcontext():
            if slot.compute_read_done is not None:
                self.stream.wait_event(slot.compute_read_done)
            gate_up, down = executor.read_weights()
            self._ensure_slots(gate_up, down)
            prepared = slot.home_layout_key == key
            source_gate_up, source_down = (slot.home_gate_up, slot.home_down) if prepared else (gate_up, down)
            sending = sum(plan.weight_send) > 0
            # Only senders need normalized inputs before entering the collective.
            # Receivers can cover P's remaining work with the incoming transfer.
            if prepared and self.stream is not None:
                # P owns the scratch; protect W's readers even if a later launch fails.
                source_gate_up.record_stream(self.stream)
                source_down.record_stream(self.stream)
            if prepared and sending and self.stream is not None:
                self.stream.wait_event(slot.home_layout_ready)
            with torch.profiler.record_function("moonep::ep_weight_exchange"):
                remote_gate_up, remote_down, keepalive = self._exchange_weights(
                    executor, plan, source_gate_up if sending else gate_up, source_down if sending else down)
            if prepared and not sending and self.stream is not None:
                self.stream.wait_event(slot.home_layout_ready)
            active = len(plan.group_counts)
            target_gate_up, target_down = slot.gate_up[:active], slot.down[:active]
            if not (self.compact_weight_slots and prepared):
                target_gate_up[:plan.home_count].copy_(source_gate_up)
                target_down[:plan.home_count].copy_(source_down)
            if sum(plan.weight_recv):
                target_gate_up[plan.home_count:].copy_(remote_gate_up)
                target_down[plan.home_count:].copy_(remote_down)
            ready = None
            if self.stream is not None:
                if not prepared:
                    gate_up.record_stream(self.stream)
                    down.record_stream(self.stream)
                ready = torch.npu.Event()
                ready.record(self.stream)
            if prepared:
                slot.home_layout_read_done = ready
        # Exchange scratch has only EP readers, all ordered before ready.
        # Retaining it until C consumes the table would pin prefetch-depth copies.
        del keepalive
        if getattr(self, "release_w_workspace", False):
            self.weight_workspace.clear()
        lease = _WeightLease(slot, target_gate_up, target_down, ready)
        self.weight_workspace_busy = False
        self.weight_leases[key] = lease
        self.statistics["materializations"] += 1
        self.statistics["max_live_weight_leases"] = max(
            self.statistics["max_live_weight_leases"], len(self.weight_leases))
        return lease

    def _prepare_weight_views(self, key: ExecutionKey) -> _WeightLease:
        """Borrow homes and materialize only remote matrices in their physical layout."""
        executor, plan = self.executions[key]
        gate_up, down = executor.read_weights()
        if self.stream is None and gate_up.device.type == "npu":
            self.stream = communication_stream(executor.weight_group, gate_up.device)
        current = torch.npu.current_stream(gate_up.device) if self.stream is not None else None
        if self.weight_workspace_busy:
            raise RuntimeError("weight workspace still belongs to an incomplete materialization")
        self.weight_workspace_busy = self.stream is not None
        received = sum(plan.weight_recv)
        capacity = min(executor.replica_capacity, ((received + 3) // 4) * 4)
        slot = next((item for item in self.slots if item.key is None), None)
        context = torch.npu.stream(self.stream) if self.stream is not None else nullcontext()
        with context:
            if self.stream is not None:
                self.stream.wait_stream(current)
            if slot is not None and slot.compute_read_done is not None and self.stream is not None:
                self.stream.wait_event(slot.compute_read_done)
            if slot is not None and any(target.shape[1:] != source.shape[1:] or target.dtype != source.dtype
                                         or target.device != source.device
                                         for target, source in ((slot.gate_up, gate_up), (slot.down, down))):
                raise ValueError("shared weight views require uniform expert dimensions, dtype, and device")
            if slot is None or slot.gate_up.shape[0] < received:
                matrices = []
                for value in (gate_up, down):
                    transposed = value.stride(-2) == 1
                    shape = tuple(reversed(value.shape[1:])) if transposed else tuple(value.shape[1:])
                    buffer = value.new_empty((capacity, *shape))
                    matrices.append(buffer.transpose(1, 2) if transposed else buffer)
                if slot is None:
                    slot = _WeightTableSlot(matrices[0], matrices[1])
                    self.slots.append(slot)
                else:
                    slot.gate_up, slot.down = matrices
            slot.key = key
            if self.stream is not None and sum(plan.weight_send):
                gate_up.record_stream(self.stream)
                down.record_stream(self.stream)
            remote_gate, remote_down, keepalive = self._exchange_weights(executor, plan, gate_up, down)
            if received:
                slot.gate_up[:received].copy_(remote_gate)
                slot.down[:received].copy_(remote_down)
            ready = None
            if self.stream is not None:
                ready = torch.npu.Event()
                ready.record(self.stream)
        del keepalive
        self.weight_workspace_busy = False
        if self.release_w_workspace:
            self.weight_workspace.clear()
        if getattr(self, "home_weight_mode", "1") == "2":
            rows = (sum(plan.group_counts[:plan.home_count]), sum(plan.group_counts[plan.home_count:]))
            weights = tuple(GroupedWeightViews(parts=(home, remote[:received]), rows_per_part=rows)
                            for home, remote in ((gate_up, slot.gate_up), (down, slot.down)))
        else:
            weights = tuple(GroupedWeightViews(tuple(home.unbind(0)) + tuple(remote[:received].unbind(0)))
                            for home, remote in ((gate_up, slot.gate_up), (down, slot.down)))
        lease = _WeightLease(slot, *weights, ready)
        self.weight_leases[key] = lease
        self.statistics["materializations"] += 1
        self.statistics["max_live_weight_leases"] = max(
            self.statistics["max_live_weight_leases"], len(self.weight_leases))
        return lease

    def _borrow_weight_workspace(self, role: str, template: torch.Tensor,
                                 elements: int, limit: int) -> torch.Tensor:
        """按需建立有界工作区，在唯一预取流复用，超限使用独立临时存储。

        Args:
            role: 发送或接收工作区；FSDP 自行管理 AG 转换输入。
            template: 决定dtype和device的已有权重。
            elements: 当前规划实际需要的元素数。
            limit: 由专家数与副本容量配置推导的工作区预算，不代表立即申请量。

        Returns:
            足够容纳当前请求的连续视图；超预算请求不改变已有工作区。

        Example:
            预算16行，首次接收1行只申请1行；下次接收3行增长为3行，
            再接收4行增长为6行，后续不超过6行均复用。17行走独立申请。
            步末释放Tensor但记住实际需求4行，下一步首次接收1行直接申请4行。
            所有读写在W按home就绪→打包→A2Av wait→填表顺序提交；
            增长创建新storage，已有权重借用保留的旧视图不会被resize破坏。
        """
        if elements == 0 or elements > limit:
            if elements > limit:
                self.statistics["weight_workspace_oversized"] += 1
            return template.new_empty(elements)
        demand = (elements if getattr(self, "release_w_workspace", False)
                  else max(elements, min(limit, self.weight_workspace_demand.get(role, 0))))
        self.weight_workspace_demand[role] = demand
        buffer = self.weight_workspace.get(role)
        if buffer is not None and (buffer.dtype != template.dtype or buffer.device != template.device
                                   or buffer.numel() > limit):
            raise ValueError("共享权重工作区要求相同的dtype、device且不超过配置预算")
        if buffer is None or buffer.numel() < elements:
            capacity = demand if buffer is None else min(limit, max(demand, 2 * buffer.numel()))
            buffer = template.new_empty(capacity)
            self.weight_workspace[role] = buffer
            self.statistics["weight_workspace_allocations"] += 1
            self.statistics["weight_workspace_peak_bytes"] = max(
                self.statistics["weight_workspace_peak_bytes"],
                sum(value.numel() * value.element_size() for value in self.weight_workspace.values()),
            )
        return buffer[:elements]

    def _exchange_weights(self, executor: Any, plan: ExecutionPlan, gate_up: torch.Tensor,
                          down: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, list]:
        """打包两矩阵交换，临时缓冲保留到ready；例如e0的两组权重只发一次A2Av。"""
        if not plan.any_migration:
            return None, None, []
        gate_up_width, down_width = gate_up.shape[1] * gate_up.shape[2], down.shape[1] * down.shape[2]
        sent, received = sum(plan.weight_send), sum(plan.weight_recv)
        width = gate_up_width + down_width
        packed = None
        if self.stream is not None:
            # Keep transient sends in the stream that packs and consumes them,
            # including oversized packets that bypass the bounded workspace.
            limit = self.weight_send_capacity * width
            packed = self._borrow_weight_workspace("send", gate_up, sent * width, limit).view(sent, width)
            limit = self.weight_recv_capacity * width
            remote = self._borrow_weight_workspace("recv", gate_up, received * width, limit).view(received, width)
        else:
            remote = gate_up.new_empty((received, width))
        packed = pack_expert_rows(gate_up, down, plan.send_ids, out=packed)
        # 精确包住 collective 提交；在 ASCEND_PROFILER_OUTPUT 的 CPU 轨道中，
        # HcclAlltoAllV 会嵌套于此标记，并通过 HostToDevice 箭头对应设备通信。
        with torch.profiler.record_function("moonep::all_to_all_v::weight_replica"):
            work = dist.all_to_all_single(remote, packed, plan.weight_recv, plan.weight_send,
                                         group=executor.weight_group, async_op=True)
        # 后续组表马上要在 W 流读取 remote，因此此处只建立 W 对 HCCL 的依赖。
        # 主计算流不在这里等待；它直到 consume() 紧邻专家 GMM 时才等待 ready。
        work.wait()
        self.statistics["weight_exchanges"] += 1
        remote_gate_up = remote[:, :gate_up_width].reshape(received, *gate_up.shape[1:])
        remote_down = remote[:, gate_up_width:].reshape(received, *down.shape[1:])
        return remote_gate_up, remote_down, [packed, remote]

    def consume(self, key: ExecutionKey) -> tuple[GroupedWeights, GroupedWeights]:
        """在真正消费流等待当前ready；例如已预取K6无需再执行AG或EP。"""
        lease = self.require_weight_lease(key)
        if self.stream is not None:
            current = torch.npu.current_stream()
            current.wait_event(lease.ready)
            # 表在 W 分配、由 C 读取；等待 ready 保证数据就绪，record_stream 保护存储生命周期。
            lease.gate_up.record_stream(current)
            lease.down.record_stream(current)
        lease.consumed = True
        return lease.gate_up, lease.down

    def release(self, key: ExecutionKey) -> None:
        """最后一次GMM读取排队后退役，下一写者等事件；例如K7槽供K5复用。"""
        lease = self.weight_leases[key]
        if not lease.consumed:
            raise RuntimeError("不能按正常路径归还尚未消费的预取")
        if self.stream is not None:
            lease.slot.compute_read_done = torch.npu.Event()
            lease.slot.compute_read_done.record(torch.npu.current_stream())
        lease.slot.key = None
        lease.slot.home_layout_key = None
        del self.weight_leases[key]

    def prefetch_previous(self, key: ExecutionKey) -> None:
        """无 FSDP 的脚本路径预取前一层；例如独立三层模型的 K7 预取 K6。"""
        if not self.trace or self.trace[-1] != key:
            raise RuntimeError("首版MoonEP预取要求按原前向调用的逆序执行backward")
        if self.standalone_prefetch and len(self.trace) > 1:
            with torch.profiler.record_function("moonep::prefetch_previous"):
                self.prepare(self.trace[-2])

    def prefetch_key(self, executor: Any) -> ExecutionKey | None:
        """Select a retained invocation only when its layer has no ambiguity."""
        if executor.layer in self.shared_executors:
            self.statistics["prefetch_ambiguous"] += 1
            return None
        return next((item for item in reversed(self.trace)
                     if item.layer == executor.layer and self.executions[item][0] is executor), None)

    def complete(self, key: ExecutionKey) -> None:
        """真实反向已提交后删除小计划；例如R7退出不会调用此函数，B7才会。"""
        executor, plan = super().complete(key)
        chunked = getattr(executor, "chunked", None)
        if chunked is not None:
            chunked.forget(plan)
        self.active_calls[key.layer] -= 1
        if not self.active_calls[key.layer]:
            del self.active_calls[key.layer]
            self.shared_executors.discard(key.layer)

    def finish_step(self) -> None:
        """优化器更新前检查租约全部归还；例如两个微批均反向后才推进版本。"""
        if (self.trace or self.executions or self.weight_leases or self.forward_inputs or self.active_calls
                or self.shared_executors or self.home_layout_slots):
            raise RuntimeError("尚有MoonEP前向/预取未完成，拒绝进入下一参数版本")
        if any(executor.gradient_extension and executor.gradient_extension.pending for executor in self.executors):
            raise RuntimeError("专家副本梯度尚未在 FSDP 归约前完成归并")
        # 池只在W使用，通信Work与record_stream保护最后读取；不resize单个借用切片。
        self.weight_workspace.clear()
        self.version += 1


def _selected_expert_rows(value: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """仅物化被选中的转置专家；例如从16个转置专家中选1个，避免先复制全部16个。

    输入三维GMM权重与专家行号，输出保留GMM逻辑维度的选中行。
    发送行数达到源专家数时使用原gather，避免重复专家的转置工作随发送量放大。
    """
    if rows.numel() < value.shape[0] and not value.is_contiguous():
        original = value.transpose(1, 2)
        if original.is_contiguous():
            return original.index_select(0, rows).transpose(1, 2)
    return value.index_select(0, rows)


@torch.no_grad()
def pack_expert_rows(gate_up: torch.Tensor, down: torch.Tensor, rows: torch.Tensor | None,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """按专家行打包两矩阵，依据源布局避免额外的全专家转置。

    输入 gate[A,H,2I]、down[A,I,H] 与 rows[S]；输出 [S,3HI]。
    例如 rows=[2,0,2] 时每行依次放该专家的 gate/up 和 down，允许重复发送。
    单行连续源直接写入连续out；多行采用gather+cat，避免NPU非连续out的内部临时量。
    可选out为连续[S,3HI]存储；例如复用EP流上的发送工作区，避免重复分配。
    rows为None时直接打包全部物理行；固定的副本梯度后缀不再gather。
    """
    count = gate_up.shape[0] if rows is None else rows.numel()
    gate_up_width, down_width = gate_up.shape[1] * gate_up.shape[2], down.shape[1] * down.shape[2]
    if out is not None and (out.shape != (count, gate_up_width + down_width) or not out.is_contiguous()
                            or out.dtype != gate_up.dtype or out.device != gate_up.device):
        raise ValueError("专家打包out必须具有正确shape、dtype、device并且连续")
    if rows is None:
        return torch.cat((gate_up.reshape(count, gate_up_width), down.reshape(count, down_width)), dim=1, out=out)
    if count == 1 and gate_up.is_contiguous() and down.is_contiguous():
        packed = gate_up.new_empty((count, gate_up_width + down_width)) if out is None else out
        gate_view = packed[:, :gate_up_width].view(count, *gate_up.shape[1:])
        down_view = packed[:, gate_up_width:].view(count, *down.shape[1:])
        torch.index_select(gate_up, 0, rows, out=gate_view)
        torch.index_select(down, 0, rows, out=down_view)
        return packed
    return torch.cat((_selected_expert_rows(gate_up, rows).reshape(count, gate_up_width),
                      _selected_expert_rows(down, rows).reshape(count, down_width)), dim=1, out=out)


@torch.no_grad()
def local_home_gradients(gate_up: torch.Tensor, down: torch.Tensor,
                         plan: ExecutionPlan) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract home gradients from the fixed physical prefix without gathering.

    返回 [P,H,2I]、[P,I,H]，未在本卡计算的 home 专家写零。
    远程贡献在副本梯度回传后执行相同的 BF16 index_add。
    """
    home_gate_up = gate_up.new_zeros((plan.home_count, *gate_up.shape[1:]))
    home_down = down.new_zeros((plan.home_count, *down.shape[1:]))
    # Retain the original zero-plus-gradient operation, including signed-zero
    # behavior; each fixed home slot contributes exactly once before remote sums.
    home_gate_up.add_(gate_up[:plan.home_count])
    home_down.add_(down[:plan.home_count])
    return home_gate_up, home_down


@torch.no_grad()
def return_home_gradients(gate_up: torch.Tensor, down: torch.Tensor, plan: ExecutionPlan,
                          group: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """EP副本梯度求和回归属，不改变FSDP缩放；例如rank1的e0 dW返回rank0。"""
    home_gate_up, home_down, packet = launch_home_gradients(gate_up, down, plan, group, None)
    if packet is not None:
        remote_gate_up, remote_down = wait_home_gradients(packet)
        home_gate_up.index_add_(0, plan.send_ids, remote_gate_up)
        home_down.index_add_(0, plan.send_ids, remote_down)
    return home_gate_up, home_down


@torch.no_grad()
def launch_home_gradients(gate_up: torch.Tensor, down: torch.Tensor, plan: ExecutionPlan,
                          group: Any, stream: Any) -> tuple[torch.Tensor, torch.Tensor, _GradientPacket | None]:
    """返回本地梯度并在现有梯度域流提交 EP 回传；例如 B7 先发出 e0 的 BF16 dW。

    Args:
        gate_up: GMM 产生的 gate/up 专家梯度。
        down: GMM 产生的 down 专家梯度。
        plan: 原前向冻结的专家副本与归属计划。
        group: 专家梯度回传的 EP 通信域。
        stream: 该通信域已有的 collective 流；CPU 测试时为 None。

    Returns:
        可以先交给 autograd 的两份本地 home 梯度，以及等待 FSDP 归并的通信包。
    """
    home_gate_up, home_down = local_home_gradients(gate_up, down, plan)
    packet = launch_replica_gradients(gate_up[plan.home_count:], down[plan.home_count:], plan, group, stream)
    return home_gate_up, home_down, packet


@torch.no_grad()
def launch_replica_gradients(gate_up: torch.Tensor, down: torch.Tensor, plan: ExecutionPlan,
                             group: Any, stream: Any) -> _GradientPacket | None:
    """Return only replica dW; home accumulators never enter the communication payload."""
    if not plan.any_migration:
        return None
    sent = sum(plan.weight_send)
    gate_up_width = gate_up.shape[1] * gate_up.shape[2]
    down_width = down.shape[1] * down.shape[2]
    packed = pack_expert_rows(gate_up, down, None)
    returned = packed.new_empty((sent, gate_up_width + down_width))
    if stream is not None:
        # G 必须在 C 完成打包后读取；通信使用梯度域既有流，避免新增流割裂分配池。
        stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream) if stream is not None else nullcontext():
        with torch.profiler.record_function("moonep::all_to_all_v::gradient_replica"):
            work = dist.all_to_all_single(returned, packed, plan.weight_send, plan.weight_recv,
                                         group=group, async_op=True)
    packet = _GradientPacket(returned, packed, plan, gate_up.shape, down.shape, work)
    return packet


@torch.no_grad()
def wait_home_gradients(packet: _GradientPacket) -> tuple[torch.Tensor, torch.Tensor]:
    """等待梯度域结果并拆出两张专家表；例如 B7 的远程 dW 在 RS 前变为可读。

    Args:
        packet: 待归并队列保活的 BF16 通信包。

    Returns:
        按原 GMM 布局排列的 gate/up 与 down 远程贡献。
    """
    packet.work.wait()
    if packet.receive_buffer.device.type != "cpu":
        stream = torch.get_device_module(packet.receive_buffer.device).current_stream(packet.receive_buffer.device)
        packet.send_buffer.record_stream(stream)
        packet.receive_buffer.record_stream(stream)
    sent = sum(packet.plan.weight_send)
    width = packet.gate_up_shape[1] * packet.gate_up_shape[2]
    gate_up = packet.receive_buffer[:, :width].reshape(sent, *packet.gate_up_shape[1:])
    down = packet.receive_buffer[:, width:].reshape(sent, *packet.down_shape[1:])
    return gate_up, down


class _ExpertFunction(torch.autograd.Function):
    """只保存激活，不将临时完整专家表挂到原图；例如F7后可立即复用其槽。"""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, home_gate_up: torch.Tensor, home_down: torch.Tensor,
                key: ExecutionKey, runtime: MoonEPRuntime) -> torch.Tensor:
        """输出专家主序结果；例如输入[C,H]经过两次GMM仍为[C,H]。"""
        executor, plan = runtime.executions[key]
        gate_up, down = runtime.consume(key)
        with torch.profiler.record_function("moonep::expert_forward"):
            output, saved = expert_forward_values(x, gate_up, down, plan.boundaries,
                                                 executor.limit, grouped_values, executor.activation)
        ctx.save_for_backward(saved.x, saved.preact, saved.intermediate, saved.boundaries)
        ctx.key, ctx.runtime = key, runtime
        ctx.shapes = home_gate_up.shape, home_down.shape
        if not is_replaying():
            runtime.release(key)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """先解包并完成必要重放，再用已恢复权重计算四GMM及梯度归属。"""
        runtime, key = ctx.runtime, ctx.key
        executor, plan = runtime.executions[key]
        x, preact, intermediate, boundaries = ctx.saved_tensors
        scheduler = getattr(executor.owner, "hsdp_scheduler", None)
        if executor.layer not in runtime.fsdp_layers:
            runtime.prepare(key)
        if scheduler is None:
            runtime.prefetch_previous(key)
        gate_up, down = runtime.consume(key)
        saved = ExpertSaved(x, preact, intermediate, boundaries, executor.limit, executor.activation)
        home_gradients = None

        def launch_replica_gradients(gate_gradient: torch.Tensor, down_gradient: torch.Tensor) -> None:
            """在最后一次 dX GMM 前提交 EP 回传；例如 B7 用 dX 计算掩盖 e0 梯度通信。"""
            nonlocal home_gradients
            with torch.profiler.record_function("moonep::replica_grad_launch"):
                home_gradients = executor.gradient_extension.start(gate_gradient, down_gradient, plan)

        with torch.profiler.record_function("moonep::expert_backward"):
            dx, dg, dd = expert_backward_values(
                grad_output, saved, gate_up, down, grouped_values, grouped_weight_grad,
                on_weight_gradients=launch_replica_gradients if scheduler is not None else None,
            )
        runtime.release(key)
        if scheduler is None:
            with torch.profiler.record_function("moonep::return_gradients"):
                hg, hd = return_home_gradients(dg, dd, plan, executor.gradient_group)
        else:
            if home_gradients is None:
                raise RuntimeError("expert backward did not publish complete home gradients")
            hg, hd = home_gradients
        if (hg.shape, hd.shape) != ctx.shapes:
            raise RuntimeError("MoonEP完整归属梯度与真实参数输入形状不一致")
        runtime.complete(key)
        return dx, hg, hd, None, None


class _MoEFunction(torch.autograd.Function):
    """Own dense token exchanges so complete dW can overlap the inverse dispatch."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, home_gate_up: torch.Tensor, home_down: torch.Tensor,
                key: ExecutionKey, runtime: MoonEPRuntime, source_rows: torch.Tensor) -> torch.Tensor:
        """Return selection outputs; router weighting remains in the existing outer graph."""
        executor, plan = runtime.executions[key]
        payload = hidden.reshape(-1, hidden.shape[-1]).index_select(0, source_rows)
        with torch.profiler.record_function("moonep::all_to_all_v::token_dispatch"):
            received = (ep_all_to_all(payload, plan.token_send, plan.token_recv, executor.group)
                        if executor.ranks > 1 else payload)
        x = received.index_select(0, plan.sort_order)
        gate_up, down = runtime.consume(key)
        with torch.profiler.record_function("moonep::expert_forward"):
            output, saved = expert_forward_values(x, gate_up, down, plan.boundaries,
                                                 executor.limit, grouped_values, executor.activation)
        ctx.save_for_backward(saved.x, saved.preact, saved.intermediate, saved.boundaries, source_rows)
        ctx.key, ctx.runtime = key, runtime
        ctx.shapes, ctx.input_shape = (home_gate_up.shape, home_down.shape), hidden.shape
        if not is_replaying():
            runtime.release(key)
        restored = torch.empty_like(output)
        restored[plan.sort_order] = output
        with torch.profiler.record_function("moonep::all_to_all_v::token_return"):
            return (ep_all_to_all(restored, plan.token_recv, plan.token_send, executor.group)
                    if executor.ranks > 1 else restored)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """Submit dX return before two full-K dW GMMs, retaining FSDP gradient ownership."""
        runtime, key = ctx.runtime, ctx.key
        x, preact, intermediate, boundaries, source_rows = ctx.saved_tensors
        executor, plan = runtime.executions[key]
        scheduler = getattr(executor.owner, "hsdp_scheduler", None)
        if executor.layer not in runtime.fsdp_layers:
            runtime.prepare(key)
        if scheduler is None:
            runtime.prefetch_previous(key)
        gate_up, down = runtime.consume(key)
        with torch.profiler.record_function("moonep::all_to_all_v::token_return_backward"):
            received = (ep_all_to_all(grad_output.contiguous(), plan.token_send, plan.token_recv, executor.group)
                        if executor.ranks > 1 else grad_output)
        derivative = received.index_select(0, plan.sort_order)
        with torch.profiler.record_function("moonep::input_gradients"):
            mid = grouped_values(derivative, down.transpose(-1, -2), boundaries)
            du = activation_gradient(mid, preact, executor.limit, executor.activation)
            del mid, preact
            expert_dx = grouped_values(du, gate_up.transpose(-1, -2), boundaries)
            # The private dV receive buffer has no readers after the gather.
            # Reuse it for dX packing without overwriting an autograd input.
            packed = received if executor.ranks > 1 else torch.empty_like(expert_dx)
            packed.zero_()
            packed.index_add_(0, plan.sort_order, expert_dx)
        returned = expert_dx if expert_dx.shape[0] == sum(plan.token_send) else packed.new_empty(
            (sum(plan.token_send), packed.shape[-1]))
        ready, work = None, None
        stream = runtime.gradient_stream(executor, packed.device) if executor.ranks > 1 else None
        if executor.ranks > 1 and stream is not None:
            stream.wait_stream(torch.npu.current_stream(packed.device))
            with torch.npu.stream(stream), torch.profiler.record_function(
                    "moonep::all_to_all_v::token_dispatch_backward"):
                work = dist.all_to_all_single(returned, packed, plan.token_send, plan.token_recv,
                                              group=executor.group, async_op=True)
                work.wait()
                packed.record_stream(stream)
                returned.record_stream(stream)
                ready = torch.npu.Event()
                ready.record(stream)
            returned.record_stream(torch.npu.current_stream(packed.device))
        else:
            with torch.profiler.record_function("moonep::all_to_all_v::token_dispatch_backward"):
                returned = (ep_all_to_all(packed, plan.token_recv, plan.token_send, executor.group)
                            if executor.ranks > 1 else packed)
        del expert_dx, received, packed
        with torch.profiler.record_function("moonep::weight_gradients"):
            fusion = scheduler is not None and executor.gradient_extension.uses_accumulated_grad(scheduler.hsdp_state)
            if fusion:
                extension = executor.gradient_extension
                targets = extension.accumulation_buffers(scheduler.hsdp_state)
                home_rows = sum(plan.group_counts[:plan.home_count])
                for target, (_parameter, transposed), left, right in zip(
                        targets, extension._layouts(scheduler.hsdp_state), (x, intermediate), (du, derivative)):
                    grouped_weight_grad_add(target, left[:home_rows], right[:home_rows],
                                            boundaries[:plan.home_count], transposed=transposed)
                replica_boundaries = boundaries[plan.home_count:] - home_rows
                if replica_boundaries.numel():
                    dg = grouped_weight_grad(x[home_rows:], du[home_rows:], replica_boundaries)
                    dd = grouped_weight_grad(intermediate[home_rows:], derivative[home_rows:], replica_boundaries)
                else:
                    dg = x.new_empty((0, *gate_up.shape[1:]))
                    dd = x.new_empty((0, *down.shape[1:]))
                stream = runtime.gradient_stream(executor, x.device) if plan.any_migration else None
                packet = launch_replica_gradients(dg, dd, plan, executor.gradient_group, stream)
                if packet is not None:
                    extension.pending.append(packet)
                    runtime.statistics["gradient_exchanges"] += 1
                runtime.statistics["home_wgrad_add_calls"] += 2
                hg = hd = None
            else:
                dd = grouped_weight_grad(intermediate, derivative, boundaries)
                dg = grouped_weight_grad(x, du, boundaries)
                if scheduler is None:
                    hg, hd = return_home_gradients(dg, dd, plan, executor.gradient_group)
                else:
                    hg, hd = executor.gradient_extension.start(dg, dd, plan)
        runtime.release(key)
        if ready is not None:
            # This event covers token return only, before the later replica G.
            torch.npu.current_stream(returned.device).wait_event(ready)
        dx = returned.new_zeros(ctx.input_shape).reshape(-1, returned.shape[-1])
        dx.index_add_(0, source_rows, returned)
        if not fusion and (hg.shape, hd.shape) != ctx.shapes:
            raise RuntimeError("MoonEP完整归属梯度与真实参数输入形状不一致")
        runtime.complete(key)
        if fusion:
            executor.gradient_extension.notify_accumulated(scheduler)
        return dx.reshape(ctx.input_shape), hg, hd, None, None, None


class MoonEPExecutor:
    """接收原始router结果，在实际experts模块边界内执行脚本MoonEP。"""

    def __init__(self, owner: torch.nn.Module, read_weights: Callable, *, ep_group: Any,
                 ep_size: int, num_experts: int, top_k: int, replica_capacity: int | None = None,
                 activation: str = "clipped", swiglu_limit: float = 0.0,
                 prefetch_depth: int = 1,
                 token_dedup: bool = False,
                 forward_ratio: float | None = None,
                 backward_ratio: float | None = None,
                 gradient_accumulation_fusion: bool | None = None) -> None:
        """绑定布局读取器；例如 EP=4 时按规划同步分发并回传 token。"""
        if ep_size < 1 or num_experts < 1 or num_experts % ep_size or not 0 < top_k <= num_experts:
            raise ValueError("非法EP、专家数或top-k")
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.owner, self.read_weights = owner, read_weights
        self.group, self.ranks, self.experts, self.top_k = ep_group, ep_size, num_experts, top_k
        self.rank = dist.get_rank(ep_group) if ep_size > 1 else 0
        self.home_count = num_experts // ep_size
        self.replica_capacity = self.home_count if replica_capacity is None else replica_capacity
        if (isinstance(self.replica_capacity, bool) or not isinstance(self.replica_capacity, Integral)
                or self.replica_capacity < 0):
            raise ValueError("副本容量必须是非负整数")
        self.activation, self.limit = activation, swiglu_limit
        self.prefetch_depth = prefetch_depth
        if not isinstance(token_dedup, bool):
            raise ValueError("token_dedup must be a boolean")
        self.token_dedup = token_dedup
        self.forward_ratio = 0.75 if forward_ratio is None else float(forward_ratio)
        self.backward_ratio = 1.0 if backward_ratio is None else float(backward_ratio)
        if not (0.0 <= self.forward_ratio <= 1.0 and 0.0 <= self.backward_ratio <= 1.0):
            raise ValueError("MoonEP dedup ratios must be between zero and one")
        self.runtime = None
        self.layer = -1
        self.weight_group, self.gradient_group = ep_group, ep_group
        self.gradient_extension = None
        self.chunked = None

    def _plan(self, ids: torch.Tensor, key: ExecutionKey | None = None) -> ExecutionPlan:
        """只为原始前向规划；例如重算命中原计划，免除再次同步小表。"""
        with torch.profiler.record_function("moonep::plan::counts"):
            valid_extent = ids.numel() <= 2**20 and self.experts <= 2**20
            if valid_extent:
                # Fixed bins avoid scatter-add's contended atomics. At this
                # bound both bin labels and counts are exactly representable.
                local = torch.histc(ids.float(), bins=self.experts,
                                    min=-0.5, max=self.experts - 0.5).long()
            else:
                local = torch.zeros(self.experts, device=ids.device, dtype=torch.int64)
            payload = torch.cat((local, local.new_full((1,), ids.numel() if valid_extent else -1)))
            if self.ranks > 1:
                gathered = payload.new_empty(self.ranks * payload.numel())
                work = dist.all_gather_into_tensor(gathered, payload, group=self.group, async_op=True)
                # This sort needs only local IDs. Submit it before waiting for
                # the remote histogram, then retain its exact stable order.
                entry_order = prepare_entry_order(ids, self.experts)
                if key is not None and ids.device.type == 'npu':
                    self.runtime.prepare_home_layout(self, key)
                work.wait()
                gathered = gathered.reshape(self.ranks, self.experts + 1)
            else:
                gathered = payload.unsqueeze(0)
                entry_order = prepare_entry_order(ids, self.experts)
                if key is not None and ids.device.type == 'npu':
                    self.runtime.prepare_home_layout(self, key)
            counts = gathered[:, :-1].contiguous()
            valid = (gathered[:, -1] >= 0).all() & (counts.sum(1) == gathered[:, -1]).all()
        with torch.profiler.record_function("moonep::plan::allocation"):
            geometry = None
            if ids.device.type == 'npu':
                key = ('plan', str(ids.device), self.ranks, self.experts, self.replica_capacity, self.rank)
                if key not in self.runtime.planning_graphs:
                    capacity, rank = self.replica_capacity, self.rank
                    slots = min(self.experts - self.experts // self.ranks, capacity)

                    def plan_tables(table: torch.Tensor) -> tuple[torch.Tensor, ...]:
                        """Compile placement and route tensors from the same gathered count matrix."""
                        placement, status = select_device_allocation(table, capacity)
                        route = route_geometry(table, placement, rank, slots)
                        return placement, status, *route.tensors()

                    self.runtime.planning_graphs[key] = TableGraph(plan_tables, counts)
                outputs = self.runtime.planning_graphs[key](counts)
                allocation, metrics = outputs[:2]
                geometry = RouteGeometry(*outputs[2:])
            else:
                allocation, metrics = select_device_allocation(counts, self.replica_capacity)
        self.runtime.statistics["plans"] += 1
        with torch.profiler.record_function("moonep::plan::compile"):
            plan = compile_plan(counts, allocation, ids, self.rank,
                                build_receive_order=self.chunked is None, replica_capacity=self.replica_capacity,
                                metrics=metrics, valid_routes=valid, entry_order=entry_order,
                                geometry=geometry,
                                chunk_rows=None if self.chunked is None else self.chunked.chunk_rows,
                                graphs=self.runtime.planning_graphs)
        self.runtime.statistics["max_selection_rows"] = max(
            self.runtime.statistics["max_selection_rows"], sum(plan.token_send))
        self.runtime.statistics["max_received_rows"] = max(
            self.runtime.statistics["max_received_rows"], sum(plan.token_recv))
        self.runtime.statistics["planner_fallbacks"] += int(plan.planner_fallback)
        self.runtime.statistics["moved_entries"] += plan.metrics[1]
        self.runtime.statistics["peak_copies"] = max(self.runtime.statistics["peak_copies"], plan.metrics[2])
        self.runtime.statistics["remote_experts_max"] = max(
            self.runtime.statistics["remote_experts_max"], sum(plan.weight_recv))
        return plan

    def prepare_forward(self, key: ExecutionKey, hidden: torch.Tensor,
                        ids: torch.Tensor, scores: torch.Tensor, *, expert_weights: Any = None) -> None:
        """Build the original route from the restoration hook; replay reuses it.

        Args:
            key: Invocation captured before FSDP's parameter restoration.
            hidden: Source token activations, used only for metadata dimensions.
            ids: Original Top-K expert IDs.
            scores: Original differentiable routing weights, retaining their dtype.
            expert_weights: Unused; replica materialization reads restored owner parameters.
        """
        del expert_weights
        if ids.shape != scores.shape or ids.numel() != hidden.numel() // hidden.shape[-1] * self.top_k:
            raise ValueError("MoonEP路由维度与token数不匹配")
        if key not in self.runtime.executions:
            with torch.profiler.record_function("moonep::plan"):
                try:
                    self.runtime.executions[key] = self, self._plan(ids.reshape(-1).long(), key)
                except BaseException:
                    self.runtime.cancel_home_layout(key)
                    raise
            if self.token_dedup and (self.forward_ratio > 0.0 or self.backward_ratio > 0.0):
                # Start replica materialization before the metadata exchange so W can cover it.
                self.runtime.prepare(key)
                _, base_plan = self.runtime.executions[key]
                local_error = None
                try:
                    local = prepare_local_dedup(base_plan.dispatch_order, base_plan.token_send,
                                                base_plan.token_recv, self.top_k,
                                                forward_ratio=self.forward_ratio,
                                                backward_ratio=self.backward_ratio,
                                                graphs=self.runtime.planning_graphs)
                except Exception as exc:
                    local_error = exc
                    local = failed_local_dedup(base_plan.token_send, base_plan.token_recv,
                                               device=ids.device, top_k=self.top_k)
                try:
                    dedup = finish_dedup(local, self.group, device=ids.device, width=hidden.shape[-1],
                                         backward_ratio=self.backward_ratio,
                                         score_lanes=max(1, scores.element_size() // hidden.element_size()),
                                         graphs=self.runtime.planning_graphs, execution=key,
                                         source_inverse=base_plan.unpermute_indices, target_order=base_plan.sort_order)
                except Exception as exc:
                    raise RuntimeError("nonfused dedup metadata exchange failed on an EP rank") from (
                        local_error if local_error is not None else exc)
                self.runtime.executions[key] = self, replace(base_plan, dedup=dedup)
                self.runtime.statistics["dedup_route_samples"] += 1
                self.runtime.statistics["dedup_dense_fallbacks"] += int(dedup.saved_rows == 0)

    def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor,
                 *, expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Consume hook-prepared weights and execute the chosen token backend."""
        if self.runtime is None:
            raise RuntimeError("configure_moonep must bind the expert lifecycle after FSDP wrapping")
        key = self.runtime.forward_key(self, hidden, ids, scores, expert_weights)
        _, plan = self.runtime.executions[key]
        weights = expert_weights if expert_weights is not None else self.read_weights()
        if self.chunked is not None:
            return self.chunked(hidden, scores, weights, key, self.runtime, plan)
        if plan.dedup is not None:
            received, received_scores = plan.dedup.dispatch_with_scores(
                hidden.reshape(-1, hidden.shape[-1]), scores.reshape(-1, self.top_k))
            computed = _ExpertFunction.apply(received, *weights, key, self.runtime)
            result = plan.dedup.return_weighted(computed, received_scores, hidden.numel() // hidden.shape[-1])
            if not torch.is_grad_enabled() or not computed.requires_grad:
                self.runtime.complete(key)
            self.runtime.statistics["dedup_saved_rows"] += plan.dedup.saved_rows
            return result.to(hidden.dtype).reshape_as(hidden)
        source_rows = plan.dispatch_order // self.top_k
        returned = _MoEFunction.apply(hidden, *weights, key, self.runtime, source_rows)
        # 评估没有专家反向来完成key；例如no_grad验证后继续训练不会残留旧轨迹。
        if not torch.is_grad_enabled() or not returned.requires_grad:
            self.runtime.complete(key)
        return _combine_returned_tokens(hidden, returned, scores, plan, source_rows)


def _configure_standalone_prefetch(model: torch.nn.Module, executors: list, prefetch_depth: int | None) -> bool:
    """Resolve nonfused standalone prefetch after sparse dispatch has returned."""
    # FSDP 决定何时对专家发起 AG；扩展 hook 随之触发 EP 通信，不读取预取深度。
    schedulers = [module.hsdp_scheduler for module in model.modules()
                  if getattr(module, "hsdp_scheduler", None) is not None]
    if schedulers:
        if prefetch_depth is not None:
            raise ValueError("FSDP 模式的专家预取由 AG hook 触发，不接受独立的 MoonEP 预取深度")
        standalone_prefetch = False
    else:
        if prefetch_depth is None:
            depths = {executor.prefetch_depth for executor in executors}
            if len(depths) != 1:
                raise ValueError("同一模型所有MoonEP单元必须采用相同预取配置")
            prefetch_depth = depths.pop()
        if (isinstance(prefetch_depth, bool) or not isinstance(prefetch_depth, Integral)
                or prefetch_depth < 0):
            raise ValueError("无 FSDP 路径的预取配置必须是非负整数")
        standalone_prefetch = bool(prefetch_depth)
    return standalone_prefetch


def configure_moonep(model: torch.nn.Module, *,
                     prefetch_depth: int | None = None) -> MoonEPRuntime | SparseMoonEPRuntime:
    """在模型物化与FSDP包装之后建立运行时；例如48层只按实际并发共享表槽。

    Args:
        model: 已绑定专家执行器的完整模型。
        prefetch_depth: 仅非融合 standalone 使用：0 关闭，正整数启用预取前一调用，
            None 从 executor 配置推导。FSDP 与融合模式不接受显式指定。

    Returns:
        非融合 MoonEP 或独立稀疏运行时；每次 optimizer 更新前调用 finish_step()。

    Note:
        稀疏路径要求所有 rank 在完整顺序前向后调用 runtime.backward(loss)，
        且 retain_graph=False、create_graph=False；不能直接调用 loss.backward()。
        非融合路径仍可使用普通 loss.backward()；融合运行时在所有图和流结束后由各 rank 一起调用 close()。
    """
    executors = [module._moonep_executor for module in model.modules()  # pylint: disable=protected-access
                 if isinstance(getattr(module, "_moonep_executor", None), (MoonEPExecutor, SparseMoonEPExecutor))]
    if any(executor.runtime is not None for executor in executors):
        raise ValueError("同一模型不能重复配置MoonEP运行时")
    if any(isinstance(executor, SparseMoonEPExecutor) for executor in executors):
        if not all(isinstance(executor, SparseMoonEPExecutor) for executor in executors):
            raise ValueError("一个模型不能混用稀疏融合与普通MoonEP执行器")
        if prefetch_depth is not None:
            raise ValueError("稀疏 MoonEP 不接受 prefetch_depth；FSDP 通知决定预取，无 FSDP 脚本按需恢复")
        runtime = SparseMoonEPRuntime(executors)
        model._moonep_runtime = runtime  # pylint: disable=protected-access
        return runtime
    standalone_prefetch = _configure_standalone_prefetch(model, executors, prefetch_depth)
    runtime = MoonEPRuntime(executors, standalone_prefetch=standalone_prefetch)
    model._moonep_runtime = runtime  # pylint: disable=protected-access
    return runtime


def has_moonep(module: torch.nn.Module) -> bool:
    """检查是否含生命周期执行器；例如普通EP模型返回False，保持已有构建行为。"""
    return any(isinstance(getattr(child, "_moonep_executor", None), (MoonEPExecutor, SparseMoonEPExecutor))
               for child in module.modules())
