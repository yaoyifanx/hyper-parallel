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
"""Explicit routed backward with complete owner gradients returned exactly once."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_parallel.core.multicore.modules.routed_experts.compute import (
    ExpertState, combine, combine_gradient,
)
from hyper_parallel.core.multicore.modules.routed_experts.grouped import (
    GroupExchange, prepare_group_exchange, submit_group_exchange,
)
from hyper_parallel.core.multicore.modules.routed_experts.transport import prepare_exchange, submit_prepared
from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import is_replaying
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import prepare_gradient_storage
from hyper_parallel.distributed.expert_parallel.moonep.fused.weights import GradientPayload


@dataclass
class _Phase:
    """Keep one phase's admitted storage alive through its last submitted reader."""

    incoming: Any
    compute: Any
    returned: Any
    result: torch.Tensor | None
    arena_token: Any
    score_gradient: torch.Tensor | None = None
    output_gradient: torch.Tensor | None = None
    group_metadata: torch.Tensor | None = None
    gradient: GradientPayload | None = None


def _prepare_forward(hidden: torch.Tensor, gate_up: list, down: list, intermediate: int,
                     executor: Any, plan: Any, runtime: Any, scores: torch.Tensor | None = None) -> _Phase:
    """Converge allocation failures once before either token direction starts."""
    phase, error = None, None
    token = runtime.arena.begin()
    try:
        groups = getattr(plan, 'grouped', None)
        if groups is not None:
            route = groups.dedup
            incoming = prepare_group_exchange(
                hidden.contiguous(), groups.source_forward, groups.target_forward,
                route.forward_send, route.forward_recv, groups.row_slots, runtime.workspace, phase=1,
                source_tokens=route.forward_tokens, scores=scores.reshape(-1),
                output_rows=plan.route.received_rows, scalar_rows=plan.route.received_rows,
                score_bf16=groups.score_bf16, ready=groups.receive_geometry.new_fence())

            def allocator(name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor | None:
                # V survives in saved_tensors; a reusable project slot would be overwritten by replay/another layer.
                """Keep saved expert outputs invocation-owned while reusing transient arena storage."""
                return None if name == 'project' else runtime.arena.allocate(name, shape, dtype)

            compute = plan.compute.prepare_forward(incoming.value, gate_up, down, intermediate, allocator=allocator)
            returned = prepare_group_exchange(
                compute.output, groups.target_forward, groups.source_forward,
                route.forward_recv, route.forward_send, groups.row_slots, runtime.workspace, phase=2,
                scores=incoming.scalars, output_rows=sum(route.forward_send),
                score_bf16=groups.score_bf16, dependency=compute.ready)
            phase = _Phase(incoming, compute, returned, None, token)
        else:
            group_metadata = (torch.empty((plan.route.received_rows, 8), dtype=torch.int64, device=hidden.device)
                             if executor.token_dedup else None)
            pipeline = plan.compute.forward_pipeline
            publish_each_block = (pipeline.enabled
                                  and pipeline.supports(hidden.shape[-1], intermediate, gate_up, down,
                                                        len(plan.route.boundaries)))
            if publish_each_block:
                pipeline.prepare_route_metadata(plan.route, runtime.workspace, hidden,
                                                bool(gate_up and gate_up[0].stride(-2) == 1), intermediate)
            incoming = prepare_exchange(hidden.contiguous(), plan.route, runtime.workspace,
                                        mapping=plan.order // executor.top_k,
                                        dedup=executor.token_dedup, group_metadata=group_metadata,
                                        publish_each_block=publish_each_block,
                                        forward_compact_max_unique_ratio=executor.forward_ratio)
            compute = plan.compute.prepare_forward(incoming.value, gate_up, down, intermediate,
                                                   allocator=runtime.arena.allocate)
            returned = prepare_exchange(compute.output, plan.route, runtime.workspace, reverse=True,
                                        dependency=compute.ready)
            phase = _Phase(incoming, compute, returned, hidden.new_empty(hidden.shape), token,
                           group_metadata=group_metadata)
    except Exception as exc:
        error = exc
    try:
        runtime.workspace.check_preflight(error)
    except BaseException:
        runtime.arena.end(token)
        raise
    return phase


def _prepare_backward(dy: torch.Tensor, state: ExpertState, values: torch.Tensor, scores: torch.Tensor,
                      gate_up: list, down: list, executor: Any, plan: Any, runtime: Any, shapes: tuple,
                      group_metadata: torch.Tensor | None) -> _Phase:
    """Allocate one reverse phase, reusing source dV after dispatch has consumed it."""
    phase, error = None, None
    token = runtime.arena.begin()
    try:
        dy = dy.contiguous()
        scheduler = getattr(executor.owner, "hsdp_scheduler", None)
        extension = getattr(executor, "gradient_extension", None)
        accumulated = extension is not None and extension.uses_accumulated_grad(scheduler.hsdp_state)
        direct_send = accumulated and runtime.direct_grad_send
        mapped, transposed = None, (False, False)
        received = staging = None
        if direct_send:
            received, staging = prepare_gradient_storage(
                runtime.arena.allocate, (plan.route.received_rows, dy.shape[1]),
                staging_elements=plan.gradient_staging_elements)
        if accumulated:
            outputs = extension.accumulation_buffers(scheduler.hsdp_state)
            transposed = tuple(value[1] for value in extension._layouts(scheduler.hsdp_state))
            if direct_send:
                mapped = runtime.gradient_domain.accumulation_targets(
                    plan.route.active_experts, executor.home_count, outputs,
                    plan.gradient_transfer, staging=staging)
            else:
                mapped = []
                for role, output in enumerate(outputs):
                    replica_count = sum(expert // executor.home_count != executor.rank
                                        for expert in plan.route.active_experts)
                    replicas = runtime.arena.allocate("replica_dW_" + str(role),
                                                      (replica_count, *output.shape[1:]), torch.float32)
                    replica_index = 0
                    targets = []
                    for expert in plan.route.active_experts:
                        if expert // executor.home_count == executor.rank:
                            targets.append(output[expert % executor.home_count])
                        else:
                            targets.append(replicas[replica_index])
                            replica_index += 1
                    mapped.append(tuple(targets))
                mapped = tuple(mapped)
        else:
            outputs = tuple(torch.empty(shape, dtype=torch.bfloat16, device=dy.device) for shape in shapes)
        groups = getattr(plan, 'grouped', None)
        dv = None if groups is not None else runtime.arena.allocate("selections", tuple(values.shape), values.dtype)
        ds = None if groups is not None else torch.empty_like(scores)
        if received is None:
            received, staging = prepare_gradient_storage(
                runtime.arena.allocate, (plan.route.received_rows, dy.shape[1]),
                staging_elements=plan.gradient_staging_elements)
        if groups is None:
            incoming = prepare_exchange(dv, plan.route, runtime.workspace, output=received, publish_each_block=True)
        else:
            route = groups.dedup
            incoming = prepare_group_exchange(
                dy, groups.source_backward, groups.target_backward, route.backward_send, route.backward_recv,
                groups.row_slots, runtime.workspace, phase=3, source_tokens=route.backward_tokens,
                target_scores=scores, saved=values, output=received, output_rows=plan.route.received_rows,
                scalar_rows=plan.route.received_rows, score_bf16=groups.score_bf16,
                ready=groups.receive_geometry.new_fence())
        compute = plan.compute.prepare_backward(incoming.value, state, gate_up, down,
                                                allocator=runtime.arena.allocate, pipeline=True,
                                                early_experts=tuple(expert // executor.home_count != executor.rank
                                                                    for expert in plan.route.active_experts),
                                                **({"gradient_outputs": mapped, "gradient_transposed": transposed}
                                                   if accumulated else {}))
        prepare_gradient = (runtime.gradient_domain.prepare_accumulated_payload if accumulated
                            else runtime.gradient_domain.prepare_gradient_payload)
        gradient = prepare_gradient(
            plan.route.active_experts, executor.home_count, (compute.gate_gradient, compute.down_gradient),
            plan.gradient_transfer, compute.gradient_ready, outputs=outputs, staging=staging,
            **({"direct_send": direct_send} if accumulated else {}))
        if groups is None:
            # T serializes dispatch and return, so its final write can reuse dV.
            returned = prepare_exchange(compute.output, plan.route, runtime.workspace, reverse=True,
                                        output=dv, dependency=compute.ready, reduce=group_metadata is not None,
                                        group_metadata=group_metadata,
                                        reduce_dense_ratio=executor.backward_ratio)
            result = dy.new_empty(dy.shape)
        else:
            route = groups.dedup
            returned = prepare_group_exchange(
                compute.output, groups.target_backward, groups.source_backward,
                route.backward_recv, route.backward_send, groups.row_slots, runtime.workspace, phase=4,
                scores=incoming.scalars, output_rows=sum(route.backward_send),
                scalar_rows=route.source_choices.numel(), score_bf16=groups.score_bf16, dependency=compute.ready)
            result = None
        phase = _Phase(incoming, compute, returned, result, token, ds, dy, gradient=gradient)
    except Exception as exc:
        error = exc
    try:
        runtime.workspace.check_preflight(error)
    except BaseException:
        runtime.arena.end(token)
        raise
    return phase


def _submit(exchange: Any, workspace: Any) -> Any:
    """Submit either packet protocol on the workspace's existing EP stream."""
    if isinstance(exchange, GroupExchange):
        return submit_group_exchange(exchange)
    return submit_prepared(exchange, workspace)


class SparseExpertFunction(torch.autograd.Function):
    """Own activation saves; borrow fresh weights independently in each phase."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, scores: torch.Tensor, home_gate_up: torch.Tensor,
                home_down: torch.Tensor, key: Any, runtime: Any) -> torch.Tensor:
        """Dispatch original tokens, evaluate actual experts, and combine once.

        Args:
            ctx: Autograd context retaining explicit backward state.
            hidden: Original BF16 token rows with their upstream gradient edge.
            scores: Differentiable BF16 or FP32 Top-K router weights.
            home_gate_up: FSDP-owned gate/up input for expert gradient attribution.
            home_down: FSDP-owned down input for expert gradient attribution.
            key: Planned execution identifying this layer and micro-batch.
            runtime: Shared weight, transport, and scratch lifecycle manager.

        Returns:
            Combined token rows whose history is owned by this autograd function.
        """
        # Queued native handlers retain their tensor arguments. Storage-sharing
        # aliases keep those references from retaining the checkpoint replay graph;
        # Function.apply still owns the original input edges for explicit backward.
        hidden, scores = hidden.detach(), scores.detach()
        executor, plan = runtime.executions[key]
        try:
            gate_up, down, weight_ready = runtime.consume(key)
            phase = _prepare_forward(hidden, gate_up, down, home_down.shape[1], executor, plan, runtime, scores)
            incoming = _submit(phase.incoming, runtime.workspace)
            torch.npu.current_stream(hidden.device).wait_event(weight_ready)
            _, state, _ = plan.compute.run_forward(phase.compute, incoming.ready)
            returned = _submit(phase.returned, runtime.workspace).wait()
            groups = getattr(plan, 'grouped', None)
            if groups is None:
                result = combine(returned, plan.inverse_order, scores, hidden.shape[0], executor.top_k,
                                 output=phase.result)
            else:
                route = groups.dedup
                result = route.sum_rows(returned, route.forward_tokens, hidden.shape[0],
                                        route.forward_source_members).to(hidden.dtype)
            runtime.arena.end(phase.arena_token)
            runtime.statistics["scratch_reserved_bytes"] = runtime.arena.reserved_bytes
            if groups is None:
                ctx.save_for_backward(state.x, state.preact, returned, scores, plan.order)
            else:
                ctx.save_for_backward(state.x, state.preact, phase.compute.output, phase.incoming.scalars)
            ctx.score_shape, ctx.score_dtype = scores.shape, scores.dtype
            ctx.group_metadata = phase.group_metadata
            ctx.key, ctx.runtime = key, runtime
            ctx.shapes = home_gate_up.shape, home_down.shape
            ctx.tokens = hidden.shape[0]
            if not is_replaying():
                runtime.release(key)
            # Autograd must attach output history to a different TensorImpl from
            # the buffer retained by the queued combine handler.
            return result.detach()
        except BaseException:
            runtime.poisoned = True
            raise

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, dy: torch.Tensor) -> tuple:
        """Finish both replica gradients before returning to FSDP AccumulateGrad."""
        runtime, key = ctx.runtime, ctx.key
        runtime.validate_backward(key)
        runtime.backward_layers.add(key.layer)
        try:
            # Unpacking may replay the checkpoint and restore parameter storage.
            # Bind home aliases only after that replay has completed.
            saved = ctx.saved_tensors
            executor, plan = runtime.executions[key]
            if executor.layer not in runtime.fsdp_layers:
                runtime.prepare(key)
            gate_up, down, weight_ready = runtime.consume(key)
            groups = getattr(plan, 'grouped', None)
            if groups is None:
                x, z, values, scores, order = saved
            else:
                x, z, values, scores = saved
            phase = _prepare_backward(dy, ExpertState(x, z), values, scores, gate_up, down, executor,
                                      plan, runtime, ctx.shapes, ctx.group_metadata)
            ds = phase.score_gradient
            # The reverse output aliases source dV only after T finishes reading it.
            if groups is None:
                _, ds = combine_gradient(phase.output_gradient, values, order, scores,
                                         outputs=(phase.returned.value, phase.score_gradient))
            incoming = _submit(phase.incoming, runtime.workspace)
            torch.npu.current_stream(phase.output_gradient.device).wait_event(weight_ready)
            def publish(role: str, _gradient: torch.Tensor) -> None:
                """Submit both prebound dW roles once their joint producer has been queued."""
                if role == "gate":
                    runtime.gradient_domain.submit_gradient_payload(phase.gradient)
                elif role != "down":
                    raise ValueError("gradient role must be down or gate")

            returned = None

            def submit_return() -> None:
                """Start token return before dW work can enqueue a wait on C."""
                nonlocal returned
                returned = _submit(phase.returned, runtime.workspace)

            plan.compute.run_backward(phase.compute, publish, incoming.ready, submit_return=submit_return)
            home_gate_up, home_down = runtime.gradient_domain.complete_gradients(phase.gradient)
            selections_dx = returned.wait()
            if groups is None:
                dx = combine(selections_dx, plan.inverse_order, None, ctx.tokens, executor.top_k, output=phase.result)
            else:
                route = groups.dedup
                dx = route.sum_rows(selections_dx, route.backward_tokens, ctx.tokens,
                                     route.backward_source_members).to(x.dtype)
                ds = returned.scalars.reshape(ctx.score_shape).to(ctx.score_dtype)
            runtime.arena.end(phase.arena_token)
            runtime.statistics["scratch_reserved_bytes"] = runtime.arena.reserved_bytes
            accumulated = getattr(phase.gradient, "accumulated", False)
            if not accumulated and (home_gate_up.shape, home_down.shape) != ctx.shapes:
                raise RuntimeError("complete sparse gradients must match the actual autograd parameter inputs")
            runtime.release(key)
            runtime.complete(key)
            ctx.group_metadata = None
            if accumulated:
                executor.gradient_extension.notify_accumulated(executor.owner.hsdp_scheduler)
                home_gate_up = home_down = None
            return dx, ds, home_gate_up, home_down, None, None
        except BaseException:
            runtime.poisoned = True
            raise
