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
"""Python D/C/R chunk pipeline using ordinary Torch collectives and expert GEMMs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.distributed.expert_parallel.moonep.checkpoint import is_replaying
from hyper_parallel.distributed.expert_parallel.moonep.chunked_math import source_gradients, store_rows, view_rows
from hyper_parallel.distributed.expert_parallel.moonep.chunked_plan import ChunkLayout, DeviceChunk, compile_chunks
from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    activation_gradient, expert_forward_values, grouped_values,
)
from hyper_parallel.distributed.expert_parallel.moonep.planning import ExecutionPlan
from hyper_parallel.distributed.expert_parallel.moonep.runtime import (
    MoonEPExecutor, submit_expert_weight_gradients,
)
from hyper_parallel.distributed.expert_parallel.moonep.streams import communication_stream


class _ChunkIOPool:
    """Reuse the token communication stream; all tensor computation stays on the caller."""

    def __init__(self, device: torch.device, group: Any = None) -> None:
        """Bind bounded I/O resources without adding an NPU application stream."""
        self.device = device
        self.module = None if device.type == 'cpu' else torch.get_device_module(device)
        self.token = None
        if device.type == 'npu':
            self.token = communication_stream(group, device)
        elif self.module is not None and group is not None:
            self.token = self.module.Stream(device=device)
        self._io: dict[tuple, list[torch.Tensor]] = {}
        self.last_use = None
        self._source_indices: torch.Tensor | None = None

    def io_pair(self, template: torch.Tensor, role: str, rows: int) -> list[torch.Tensor]:
        """Retain two I/O slots across ordered phases; growth leaves in-flight views intact."""
        key = role, template.dtype, template.shape[-1]
        pair = self._io.get(key)
        if pair is None or pair[0].shape[0] < rows:
            pair = [template.new_empty((rows, template.shape[-1])) for _ in range(2)]
            self._io[key] = pair
        return pair

    def source_indices(self, rows: int) -> torch.Tensor:
        """Reuse the identity prefix needed by per-entry native reverse combine.

        Args:
            rows: Complete source-entry extent, known from tensor metadata.

        Returns:
            An INT32 identity permutation whose storage is shared by successive layers.
        """
        if self._source_indices is None or self._source_indices.numel() < rows:
            self._source_indices = torch.arange(rows, dtype=torch.int32, device=self.device)
        return self._source_indices[:rows]

    def context(self) -> Any:
        """Select the token communication domain; CPU execution stays sequential."""
        return nullcontext() if self.token is None else self.module.stream(self.token)

    def record(self) -> Any:
        """Record completion of the current stream's submitted work."""
        if self.module is None:
            return None
        event = self.module.Event()
        event.record(self.module.current_stream(self.device))
        return event

    def wait(self, event: Any) -> None:
        """Order subsequent current-stream work after the supplied event."""
        if event is not None:
            self.module.current_stream(self.device).wait_event(event)

    def protect(self, *values: torch.Tensor | None) -> None:
        """Retain storage until the current stream has finished using it."""
        if self.module is not None:
            for value in values:
                if value is not None:
                    value.record_stream(self.module.current_stream(self.device))


@dataclass
class _ChunkPhaseBuffers:
    """Two receive/return slots and caller-owned source accumulators."""

    receive: list[torch.Tensor]
    returned: list[torch.Tensor]
    output: torch.Tensor
    score_gradient: torch.Tensor | None
    source_output: torch.Tensor
    cache_return: bool
    direct_receive: list[torch.Tensor | None] = field(default_factory=list)
    source_indices: torch.Tensor | None = None


def _buffers(hidden: torch.Tensor, scores: torch.Tensor, chunks: ChunkLayout,
             *, save_source: bool, pool: _ChunkIOPool,
             source_cache: torch.Tensor | None = None) -> _ChunkPhaseBuffers:
    width = hidden.shape[-1]
    sent = max((sum(chunk.spec.send_counts) for chunk in chunks), default=0)
    received = max((sum(chunk.spec.receive_counts) for chunk in chunks), default=0)
    identity = pool.source_indices(scores.numel()) if hidden.device.type == 'npu' else None
    return _ChunkPhaseBuffers(pool.io_pair(hidden, 'receive', received),
                    [] if save_source else pool.io_pair(hidden, 'returned', sent),
                    hidden.new_zeros(hidden.shape, dtype=hidden.dtype if save_source else torch.float32),
                    None,
                    source_cache if source_cache is not None else hidden.new_empty(
                        (scores.numel() if save_source else 0, width)), save_source, source_indices=identity)


def _exchange(output: torch.Tensor, value: torch.Tensor, receive: tuple[int, ...],
              send: tuple[int, ...], group: Any) -> None:
    if len(send) == 1:
        output.copy_(value)
        return
    work = dist.all_to_all_single(output, value, output_split_sizes=list(receive),
                                 input_split_sizes=list(send), group=group, async_op=True)
    # On accelerator backends this orders T after the collective; no output is read before wait.
    work.wait()


def _restore_return(value: torch.Tensor, chunk: DeviceChunk, scratch: torch.Tensor) -> torch.Tensor:
    """Gather through the cached inverse; an identity layout can send the result directly."""
    if chunk.restore_order is None:
        return value
    torch.index_select(value, 0, chunk.restore_order, out=scratch)
    return scratch


def _source_output(value: torch.Tensor, inverse: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Use the standard semantic unpermute kernel for one FP32-score source combine."""
    if value.device.type == 'npu':
        # Optional standard TorchNPU operator, independent of MoonEP native payloads.
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_moe_token_unpermute(value, inverse, scores)
    restored = value.index_select(0, inverse).reshape(*scores.shape, value.shape[-1])
    return (restored.float() * scores.float().unsqueeze(-1)).sum(-2).to(value.dtype)


def _receive_buffer(buffers: _ChunkPhaseBuffers, ordinal: int, rows: int) -> torch.Tensor:
    """Use an admitted full-arena view or the existing two-slot receive buffer."""
    direct = buffers.direct_receive[ordinal] if buffers.direct_receive else None
    return buffers.receive[ordinal % 2][:rows] if direct is None else direct


def _run(chunks: ChunkLayout, streams: _ChunkIOPool, buffers: _ChunkPhaseBuffers, group: Any,
         pack: Callable, compute: Callable, combine: Callable, submitted: Callable, phase: str) -> None:
    """Prepack once on C and overlap its chunk computations with the existing token stream."""
    ready = [None] * len(chunks)
    merged = [None, None]
    streams.wait(streams.last_use)
    with torch.profiler.record_function(f'moonep_chunked::{phase}::pack_all'):
        packed = pack()
    inputs_ready = streams.record()
    with streams.context():
        streams.wait(inputs_ready)
        streams.protect(packed, buffers.source_output)

    def dispatch(ordinal: int) -> None:
        """Publish one arrival event without depending on the previous source merge."""
        chunk = chunks[ordinal]
        with streams.context(), torch.profiler.record_function(f'moonep_chunked::{phase}::D{ordinal}'):
            received = sum(chunk.spec.receive_counts)
            payload = packed[chunk.source_begin:chunk.source_end]
            destination = _receive_buffer(buffers, ordinal, received)
            streams.protect(destination)
            _exchange(destination, payload, chunk.spec.receive_counts, chunk.spec.send_counts, group)
            ready[ordinal] = streams.record()

    def calculate(ordinal: int) -> tuple[torch.Tensor, Any]:
        """Enqueue Cq and its completion event before an independent return uses T."""
        chunk = chunks[ordinal]
        streams.wait(ready[ordinal])
        received = _receive_buffer(buffers, ordinal, sum(chunk.spec.receive_counts))
        streams.protect(received, chunk.receive_order, chunk.boundaries)
        with torch.profiler.record_function(f'moonep_chunked::{phase}::C{ordinal}'):
            value = compute(ordinal, chunk, received)
            scratch = buffers.receive[ordinal % 2][:sum(chunk.spec.receive_counts)]
            value = _restore_return(value, chunk, scratch)
        return value, streams.record()

    if chunks:
        dispatch(0)
        if len(chunks) > 1:
            dispatch(1)
        current = calculate(0)
    for ordinal, chunk in enumerate(chunks):
        # The last backward block submits complete dW and owner hooks. The preceding dX return is already independent.
        defer_last = phase == 'B' and ordinal + 2 == len(chunks)
        following = calculate(ordinal + 1) if ordinal + 1 < len(chunks) and not defer_last else None
        value, computed = current
        with streams.context(), torch.profiler.record_function(f'moonep_chunked::{phase}::R{ordinal}'):
            streams.wait(computed)
            streams.protect(value)
            if buffers.cache_return:
                returned = buffers.source_output[chunk.source_begin:chunk.source_end]
            else:
                returned = buffers.returned[ordinal % 2][:sum(chunk.spec.send_counts)]
                if returned.shape[0]:
                    streams.wait(merged[ordinal % 2])
            _exchange(returned, value, chunk.spec.send_counts, chunk.spec.receive_counts, group)
            returned_ready = streams.record()
        if defer_last:
            following = calculate(ordinal + 1)
        if ordinal + 2 < len(chunks):
            dispatch(ordinal + 2)
        if not buffers.cache_return:
            with torch.profiler.record_function(f'moonep_chunked::{phase}::M{ordinal}'):
                streams.wait(returned_ready)
                streams.protect(returned)
                if returned.shape[0]:
                    combine(chunk, returned)
                merged[ordinal % 2] = streams.record()
        current = following
    submitted()
    with streams.context():
        completed = streams.record()
    streams.wait(completed)
    if buffers.cache_return and chunks and packed.shape[0]:
        with torch.profiler.record_function(f'moonep_chunked::{phase}::source_combine'):
            combine(chunks[-1], buffers.source_output)
    streams.last_use = streams.record()
    streams.protect(buffers.output)
    if buffers.score_gradient is not None:
        streams.protect(buffers.score_gradient)


class ChunkedMoonEP:
    """Optional Python transport backend shared by sequential MoonEP expert layers.

    This implementation uses ordinary all_to_all_single and grouped GEMM. The
    token group also carries replica weights and gradients in submission order.
    """

    def __init__(self, token_group: Any, chunk_rows: int) -> None:
        """Bind the shared EP payload group and the per-round receive-row limit.

        Args:
            token_group: Shared payload group with the EP members in their original order.
            chunk_rows: Positive maximum received entries in a compute round.
        """
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise ValueError('chunk_rows must be a positive integer')
        self.group, self.chunk_rows = token_group, chunk_rows
        self._plans: dict[int, ChunkLayout] = {}
        self._streams: dict[torch.device, _ChunkIOPool] = {}

    def forget(self, plan: ExecutionPlan) -> None:
        """Release only this completed execution's immutable chunk metadata.

        Args:
            plan: Placement retired by the existing MoonEP lifecycle.
        """
        self._plans.pop(id(plan), None)

    def _prepare(self, plan: ExecutionPlan, executor: Any, *, backward_maps: bool = True) -> ChunkLayout:
        chunks = self._plans.get(id(plan))
        if chunks is None:
            chunks = compile_chunks(plan, self.chunk_rows, executor.top_k, backward_maps=backward_maps)
            self._plans[id(plan)] = chunks
        elif chunks.plan is not plan:
            raise RuntimeError('chunk cache identity differs from the retained execution')
        return chunks

    def __call__(self, hidden: torch.Tensor, scores: torch.Tensor, weights: tuple[torch.Tensor, torch.Tensor],
                 key: Any, runtime: Any, plan: ExecutionPlan) -> torch.Tensor:
        """Run one complete source-token expert operation with explicit backward.

        Args:
            hidden: Source token tensor ending in hidden_size.
            scores: Original floating TopK weights; FP32 is preserved.
            weights: Current home parameter views, maintaining their autograd edges.
            key: Original MoonEP execution key, reused by checkpoint replay.
            runtime: Existing nonfused MoonEP/FSDP weight and gradient lifecycle.
            plan: Existing CPU placement and device dispatch maps.

        Returns:
            Weighted source-token output with the same shape and dtype as hidden.
        """
        needed = torch.is_grad_enabled() and any(value.requires_grad for value in (hidden, scores, *weights))
        result = _ChunkFunction.apply(hidden, scores, *weights, key, runtime, self, plan, needed)
        if not torch.is_grad_enabled() or not result.requires_grad:
            runtime.complete(key)
        return result


def _prepare_phase(backend: ChunkedMoonEP, plan: ExecutionPlan, executor: Any,
                   hidden: torch.Tensor, scores: torch.Tensor, save_source: bool,
                   source_cache: torch.Tensor | None = None
                   ) -> tuple[_ChunkIOPool, ChunkLayout, _ChunkPhaseBuffers]:
    streams = backend._streams.get(hidden.device)
    if streams is None:
        streams = _ChunkIOPool(hidden.device, backend.group)
        backend._streams[hidden.device] = streams
    if hidden.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('chunk input must use FP32, BF16 or FP16')
    chunks = backend._prepare(plan, executor, backward_maps=save_source)
    buffers = _buffers(hidden, scores, chunks, save_source=save_source, pool=streams, source_cache=source_cache)
    return streams, chunks, buffers


class _ChunkFunction(torch.autograd.Function):
    """Split token transport and row-local GEMMs; reduce each complete expert dW once."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, scores: torch.Tensor, home_gate_up: torch.Tensor,
                home_down: torch.Tensor, key: Any, runtime: Any, backend: ChunkedMoonEP,
                plan: ExecutionPlan, needed: bool) -> torch.Tensor:
        """Save full expert inputs for training without retaining a weight lease."""
        executor, _ = runtime.executions[key]
        need_scores = needed and ctx.needs_input_grad[1]
        streams, chunks, buffers = _prepare_phase(backend, plan, executor, hidden, scores, needed)
        rows = sum(plan.token_recv)
        full_x = hidden.new_empty((rows, hidden.shape[-1])) if needed else None
        full_a = hidden.new_empty((rows, home_down.shape[1])) if needed else None
        if needed:
            buffers.direct_receive = [view_rows(full_x, chunk) if chunk.receive_order is None else None
                                      for chunk in chunks]
        source, flat_scores = hidden.reshape(-1, hidden.shape[-1]), scores.reshape(-1)
        output = buffers.output.reshape_as(source)
        saved, current_weights = [], []

        def weights() -> tuple:
            """Consume the original FSDP weight lease at the first actual reader."""
            if not current_weights:
                current_weights.extend(runtime.consume(key))
            return tuple(current_weights)

        def pack() -> torch.Tensor:
            """Gather the complete chunk-major payload once on the compute stream."""
            return source.index_select(0, chunks.source_tokens)

        def compute(_ordinal: int, chunk: DeviceChunk, received: torch.Tensor) -> torch.Tensor:
            """Run the same two expert GEMMs and activation on one row subset."""
            gate_up, down = weights()
            begin, end = chunk.spec.expert_slot_begin, chunk.spec.expert_slot_end
            if chunk.receive_order is not None and needed:
                ordered = view_rows(full_x, chunk)
                torch.index_select(received, 0, chunk.receive_order, out=ordered)
            else:
                ordered = received.index_select(0, chunk.receive_order) if chunk.receive_order is not None else received
            if needed and chunk.receive_order is None:
                ordered = store_rows(full_x, chunk, ordered)
            value, state = expert_forward_values(ordered, gate_up[begin:end], down[begin:end], chunk.boundaries,
                                                 executor.limit, grouped_values, executor.activation)
            if needed:
                store_rows(full_a, chunk, state.intermediate)
                saved.append(state.preact)
            return value

        def combine(chunk: DeviceChunk, returned: torch.Tensor) -> None:
            """Combine a complete training source cache once in original TopK order."""
            streams.protect(chunks.wire_inverse, plan.unpermute_indices)
            streams.protect(flat_scores, chunks.source_order)
            if buffers.cache_return:
                if chunk.source_end == scores.numel():
                    output.copy_(_source_output(buffers.source_output, chunks.source_order,
                                                flat_scores.reshape(-1, executor.top_k)))
            else:
                selected = flat_scores.index_select(0, chunk.source_choices).float()
                output.index_add_(0, chunk.source_tokens, returned.float() * selected[:, None])

        def submitted() -> None:
            """Retire the original forward lease after its final weight reader."""
            weights()
            if not is_replaying():
                runtime.release(key)
                if needed:
                    # Retain the source accumulation order for replay/backward;
                    # this caller-side work can overlap already submitted returns.
                    _ = chunks.source_gradient_order

        _run(chunks, streams, buffers, backend.group, pack, compute, combine, submitted, 'F')
        if needed:
            cache = buffers.source_output if need_scores else hidden.new_empty((0, hidden.shape[-1]))
            ctx.save_for_backward(scores, cache, full_x, full_a, *saved)
            ctx.key, ctx.runtime, ctx.backend, ctx.plan = key, runtime, backend, plan
            ctx.home_shapes, ctx.need_scores = (home_gate_up.shape, home_down.shape), need_scores
        return buffers.output.to(hidden.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Preserve complete dW and source accumulation order while pipelining dMid/dX."""
        runtime, key, backend, plan = ctx.runtime, ctx.key, ctx.backend, ctx.plan
        executor, _ = runtime.executions[key]
        scores, source_output, full_x, full_a, *preacts = ctx.saved_tensors
        scheduler = getattr(executor.owner, 'hsdp_scheduler', None)
        if executor.layer not in runtime.fsdp_layers:
            runtime.prepare(key)
        if scheduler is None:
            runtime.prefetch_previous(key)
        streams, chunks, buffers = _prepare_phase(
            backend, plan, executor, grad_output, scores, True,
            source_cache=source_output if ctx.need_scores else None)
        source = grad_output.reshape(-1, grad_output.shape[-1])
        full_dy = grad_output.new_empty((full_x.shape[0], source.shape[-1]))
        full_du = grad_output.new_empty((full_x.shape[0], full_a.shape[-1] * 2))
        buffers.direct_receive = [view_rows(full_dy, chunk) if chunk.receive_order is None else None
                                  for chunk in chunks]
        current_weights, home_gradients = [], []
        accumulate_home_grads = False

        def weights() -> tuple:
            """Use the original version's rematerialized weights until the final dX."""
            if not current_weights:
                current_weights.extend(runtime.consume(key))
            return tuple(current_weights)

        def launch() -> None:
            """Publish complete gradients through the existing FSDP extension."""
            nonlocal accumulate_home_grads
            with torch.profiler.record_function('moonep_chunked::weight_gradients'):
                streams.protect(chunks.canonical_order)
                values = full_x, full_a, full_dy, full_du
                gradients, accumulate_home_grads = submit_expert_weight_gradients(
                    executor, runtime, plan, values, scheduler, chunks.canonical_order)
                home_gradients.extend(gradients)

        def pack() -> torch.Tensor:
            """Prepare complete dY and score derivatives once before the first dispatch."""
            selected = source.index_select(0, chunks.source_tokens)
            probabilities = scores.reshape(-1).index_select(0, chunks.source_choices).float()[:, None]
            if ctx.need_scores and selected.shape[0]:
                weighted, derivative = source_gradients(
                    selected, source_output, probabilities, buffers.source_indices)
                buffers.score_gradient = derivative
                return weighted
            torch.mul(selected, probabilities, out=selected)
            if ctx.need_scores:
                buffers.score_gradient = scores.new_empty((0, 1))
            return selected

        def compute(ordinal: int, chunk: DeviceChunk, received: torch.Tensor) -> torch.Tensor:
            """Save row-local derivatives; each full dW sees the same rows as nonchunk execution."""
            gate_up, down = weights()
            begin, end = chunk.spec.expert_slot_begin, chunk.spec.expert_slot_end
            if chunk.receive_order is not None:
                value = view_rows(full_dy, chunk)
                torch.index_select(received, 0, chunk.receive_order, out=value)
            else:
                value = store_rows(full_dy, chunk, received)
            mid = grouped_values(value, down[begin:end].transpose(-1, -2), chunk.boundaries)
            du = activation_gradient(mid, preacts[ordinal], executor.limit, executor.activation)
            du = store_rows(full_du, chunk, du)
            if ordinal + 1 == len(chunks):
                launch()
            return grouped_values(du, gate_up[begin:end].transpose(-1, -2), chunk.boundaries)

        def combine(chunk: DeviceChunk, _returned: torch.Tensor) -> None:
            """Sum dX once in original dispatch order using only the forward unpermute op."""
            if chunk.source_end == scores.numel():
                streams.protect(chunks.source_gradient_order)
                unit_scores = torch.ones_like(scores).reshape(-1, executor.top_k)
                buffers.output.reshape_as(source).copy_(_source_output(
                    buffers.source_output, chunks.source_gradient_order, unit_scores))

        def submitted() -> None:
            """Release the weight lease after its final reader; handle globally empty input."""
            weights()
            if not chunks:
                launch()
            runtime.release(key)

        _run(chunks, streams, buffers, backend.group, pack, compute, combine, submitted, 'B')
        score_gradient = None
        if ctx.need_scores:
            streams.protect(chunks.source_order)
            score_gradient = buffers.score_gradient.reshape(-1).index_select(0, chunks.source_order).view_as(scores)
        if not accumulate_home_grads and tuple(value.shape for value in home_gradients) != ctx.home_shapes:
            raise RuntimeError('chunked owner gradients differ from the real home parameter shapes')
        runtime.complete(key)
        if accumulate_home_grads:
            executor.gradient_extension.notify_accumulated(scheduler)
        runtime.prefetch_after_backward(key, buffers.output.device)
        ctx.plan = None
        return (buffers.output.to(grad_output.dtype), score_gradient, *home_gradients,
                None, None, None, None, None)


def enable_chunked_moonep(model: torch.nn.Module, *, token_group: Any = None,
                          chunk_rows: int = 4096) -> ChunkedMoonEP:
    """Install chunks after binding expert executors and before configuring their runtime.

    Args:
        model: Model with MoonEPExecutor owners; install before configure_moonep.
        token_group: Optional existing payload group; defaults to the bound EP group.
            Token, replica-weight and replica-gradient payloads reuse this group.
        chunk_rows: Maximum received entries per round, independent of EP degree.

    Returns:
        Shared backend reusing the token domain and retaining immutable per-call maps.

    Raises:
        ValueError: Missing nonfused experts, active executions, enabled dedup,
            or a token domain with different ordered EP members.
    """
    executors = [getattr(module, '_moonep_executor', None) for module in model.modules()]
    executors = [executor for executor in executors if isinstance(executor, MoonEPExecutor)]
    if not executors:
        raise ValueError('bind nonfused MoonEP experts before enabling the Python chunk pipeline')
    group = executors[0].group if token_group is None else token_group
    backend = ChunkedMoonEP(group, chunk_rows)
    for executor in executors:
        if executor.token_dedup:
            raise ValueError('the Python chunk transport currently requires moonep_token_dedup=False')
        if executor.runtime is not None:
            raise ValueError('install the chunk backend before configure_moonep')
        if executor.ranks > 1:
            if dist.get_process_group_ranks(group) != dist.get_process_group_ranks(executor.group):
                raise ValueError('token group must preserve the ordered EP members')
    for executor in executors:
        executor.chunked = backend
    return backend
