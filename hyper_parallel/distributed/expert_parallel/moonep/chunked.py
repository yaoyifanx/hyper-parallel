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
from hyper_parallel.distributed.expert_parallel.moonep.chunked_math import ExpertArena, source_gradients
from hyper_parallel.distributed.expert_parallel.moonep.chunked_plan import ChunkLayout, DeviceChunk, compile_chunks
from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    activation_gradient, expert_forward_values, grouped_values, grouped_weight_grad,
)
from hyper_parallel.distributed.expert_parallel.moonep.planning import ExecutionPlan
from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPExecutor, return_home_gradients


class _Streams:
    """Independent token and source-merge streams; the caller owns expert computation."""

    def __init__(self, device: torch.device) -> None:
        """Create bounded token I/O and source-merge resources for one device."""
        self.device = device
        self.module = None if device.type == 'cpu' else torch.get_device_module(device)
        self.token = None if self.module is None else self.module.Stream(device=device)
        self.merge = None if self.module is None else self.module.Stream(device=device)
        self._io: dict[tuple, list[torch.Tensor]] = {}
        self.last_merge = None
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
            rows: Source entries in the largest block, known from CPU planning.

        Returns:
            An INT32 identity permutation whose storage is shared by successive layers.
        """
        if self._source_indices is None or self._source_indices.numel() < rows:
            self._source_indices = torch.arange(rows, dtype=torch.int32, device=self.device)
        return self._source_indices[:rows]

    def context(self, *, merge: bool = False) -> Any:
        """Select token or source-merge work; CPU execution stays sequential."""
        return nullcontext() if self.module is None else self.module.stream(self.merge if merge else self.token)

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
class _Buffers:
    """Two token I/O slots; the merge stream owns the source accumulators."""

    send: list[torch.Tensor]
    receive: list[torch.Tensor]
    returned: list[torch.Tensor]
    output: torch.Tensor
    score_gradient: torch.Tensor | None
    source_output: torch.Tensor
    cache_return: bool
    direct_receive: list[torch.Tensor | None] = field(default_factory=list)
    source_indices: torch.Tensor | None = None


def _buffers(hidden: torch.Tensor, scores: torch.Tensor, chunks: ChunkLayout,
             *, save_source: bool, score_gradient: bool, pool: _Streams | None = None,
             source_cache: torch.Tensor | None = None) -> _Buffers:
    width = hidden.shape[-1]
    sent = max((sum(chunk.spec.send_counts) for chunk in chunks), default=0)
    received = max((sum(chunk.spec.receive_counts) for chunk in chunks), default=0)
    def pair(role: str, rows: int) -> list[torch.Tensor]:
        """Borrow role-specific slots or allocate isolated scratch for a standalone schedule."""
        return (pool.io_pair(hidden, role, rows) if pool is not None
                else [hidden.new_empty((rows, width)) for _ in range(2)])

    identity = None
    if hidden.device.type == 'npu':
        identity = (pool.source_indices(sent) if pool is not None
                    else torch.arange(sent, dtype=torch.int32, device=hidden.device))
    return _Buffers(pair('send', sent), pair('receive', received),
                    [] if save_source else pair('returned', sent),
                    hidden.new_zeros(hidden.shape, dtype=hidden.dtype if save_source else torch.float32),
                    torch.empty(scores.shape, dtype=scores.dtype, device=scores.device) if score_gradient else None,
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


def _receive_buffer(buffers: _Buffers, ordinal: int, rows: int) -> torch.Tensor:
    """Use an admitted full-arena view or the existing two-slot receive buffer."""
    direct = buffers.direct_receive[ordinal] if buffers.direct_receive else None
    return buffers.receive[ordinal % 2][:rows] if direct is None else direct


def _run(chunks: ChunkLayout, streams: _Streams, buffers: _Buffers, group: Any,
         pack: Callable, compute: Callable, combine: Callable, submitted: Callable, phase: str) -> None:
    """Queue independent compute and returns without making the prior dX wait for full dW submission."""
    ready = [None] * len(chunks)
    merged = [None, None]
    inputs_ready = streams.record()
    with streams.context():
        # A new caller stream need not inherit C's previous join; preserve the old M reader before reusing I/O.
        streams.wait(streams.last_merge)
        streams.wait(inputs_ready)
    with streams.context(merge=True):
        streams.wait(inputs_ready)
        streams.protect(buffers.output, buffers.source_output)

    def dispatch(ordinal: int) -> None:
        """Publish one arrival event without depending on the previous source merge."""
        chunk, slot = chunks[ordinal], ordinal % 2
        with streams.context(), torch.profiler.record_function(f'moonep_chunked::{phase}::D{ordinal}'):
            sent, received = sum(chunk.spec.send_counts), sum(chunk.spec.receive_counts)
            payload = buffers.send[slot][:sent]
            if sent:
                pack(chunk, payload)
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
            streams.protect(value, chunk.restore_order)
            scratch = buffers.receive[ordinal % 2][:sum(chunk.spec.receive_counts)]
            payload = _restore_return(value, chunk, scratch)
            if buffers.cache_return:
                returned = buffers.source_output[chunk.source_begin:chunk.source_end]
            else:
                returned = buffers.returned[ordinal % 2][:sum(chunk.spec.send_counts)]
                if returned.shape[0]:
                    streams.wait(merged[ordinal % 2])
            _exchange(returned, payload, chunk.spec.send_counts, chunk.spec.receive_counts, group)
            returned_ready = streams.record()
        if defer_last:
            following = calculate(ordinal + 1)
        if ordinal + 2 < len(chunks):
            dispatch(ordinal + 2)
        with streams.context(merge=True), torch.profiler.record_function(f'moonep_chunked::{phase}::M{ordinal}'):
            streams.wait(returned_ready)
            streams.protect(returned, chunk.source_choices, chunk.source_tokens, chunk.send_indices)
            if returned.shape[0]:
                combine(chunk, returned)
            merged[ordinal % 2] = streams.record()
        current = following
    submitted()
    with streams.context():
        completed = streams.record()
    streams.wait(completed)
    with streams.context(merge=True):
        merged_ready = streams.record()
    streams.last_merge = merged_ready
    streams.wait(merged_ready)
    streams.protect(buffers.output)
    if buffers.score_gradient is not None:
        streams.protect(buffers.score_gradient)


class ChunkedMoonEP:
    """Optional Python transport backend shared by sequential MoonEP expert layers.

    This implementation uses ordinary all_to_all_single and grouped GEMM. The
    caller supplies a token group independent of weight/gradient communication.
    """

    def __init__(self, token_group: Any, chunk_rows: int) -> None:
        """Bind the independent token domain and the maximum local receive rows.

        Args:
            token_group: Process group with the original EP members in the same order.
            chunk_rows: Positive maximum received entries in a compute round.
        """
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise ValueError('chunk_rows must be a positive integer')
        self.group, self.chunk_rows = token_group, chunk_rows
        self._plans: dict[int, tuple[ExecutionPlan, ChunkLayout]] = {}
        self._streams: dict[torch.device, _Streams] = {}

    def forget(self, plan: ExecutionPlan) -> None:
        """Release only this completed execution's immutable chunk metadata.

        Args:
            plan: Placement retired by the existing MoonEP lifecycle.
        """
        self._plans.pop(id(plan), None)

    def _prepare(self, plan: ExecutionPlan, executor: Any) -> ChunkLayout:
        previous = self._plans.get(id(plan))
        if previous is None:
            chunks = compile_chunks(plan, executor.rank, self.chunk_rows, executor.top_k)
            self._plans[id(plan)] = plan, chunks
        else:
            if previous[0] is not plan:
                raise RuntimeError('chunk cache identity differs from the retained execution')
            chunks = previous[1]
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
                   score_gradient: bool, source_cache: torch.Tensor | None = None
                   ) -> tuple[_Streams, ChunkLayout, _Buffers]:
    streams = backend._streams.get(hidden.device)
    if streams is None:
        streams = _Streams(hidden.device)
        backend._streams[hidden.device] = streams
    if hidden.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError('chunk input must use FP32, BF16 or FP16')
    chunks = backend._prepare(plan, executor)
    with streams.context():
        buffers = _buffers(hidden, scores, chunks, save_source=save_source, score_gradient=score_gradient,
                           pool=streams, source_cache=source_cache)
    return streams, chunks, buffers


class _ChunkFunction(torch.autograd.Function):
    """Split token transport and row-local GEMMs; reduce each complete expert dW once."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, scores: torch.Tensor, home_gate: torch.Tensor,
                home_down: torch.Tensor, key: Any, runtime: Any, backend: ChunkedMoonEP,
                plan: ExecutionPlan, needed: bool) -> torch.Tensor:
        """Save full expert inputs for training without retaining a weight lease."""
        executor, _ = runtime.executions[key]
        need_scores = needed and ctx.needs_input_grad[1]
        streams, chunks, buffers = _prepare_phase(backend, plan, executor, hidden, scores, needed, False)
        arena = ExpertArena(chunks, plan.group_counts) if needed else None
        full_x = hidden.new_empty((arena.rows, hidden.shape[-1])) if needed else None
        full_a = hidden.new_empty((arena.rows, home_down.shape[1])) if needed else None
        if needed:
            buffers.direct_receive = arena.receive_views(full_x, chunks)
        source, flat_scores = hidden.reshape(-1, hidden.shape[-1]), scores.reshape(-1)
        output = buffers.output.reshape_as(source)
        saved, current_weights = [], []

        def weights() -> tuple:
            """Consume the original FSDP weight lease at the first actual reader."""
            if not current_weights:
                current_weights.extend(runtime.consume(key))
            return tuple(current_weights)

        def pack(chunk: DeviceChunk, payload: torch.Tensor) -> None:
            """Gather each outgoing source token directly into its T-stream slot."""
            streams.protect(source, chunk.source_tokens, chunk.source_choices, chunk.send_indices)
            torch.index_select(source, 0, chunk.source_tokens, out=payload)

        def compute(ordinal: int, chunk: DeviceChunk, received: torch.Tensor) -> torch.Tensor:
            """Run the same two expert GEMMs and activation on one row subset."""
            gate, down = weights()
            begin, end = chunk.spec.weight_begin, chunk.spec.weight_end
            ordered = received.index_select(0, chunk.receive_order) if chunk.receive_order is not None else received
            if needed:
                ordered = arena.store(full_x, ordinal, ordered)
            value, state = expert_forward_values(ordered, gate[begin:end], down[begin:end], chunk.boundaries,
                                                 executor.limit, grouped_values, executor.activation)
            if needed:
                arena.store(full_a, ordinal, state.intermediate)
                saved.append(state.preact)
            return value

        def combine(chunk: DeviceChunk, returned: torch.Tensor) -> None:
            """Combine a complete training source cache once in original TopK order."""
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

        _run(chunks, streams, buffers, backend.group, pack, compute, combine, submitted, 'F')
        if needed:
            cache = buffers.source_output if need_scores else hidden.new_empty((0, hidden.shape[-1]))
            ctx.save_for_backward(scores, cache, full_x, full_a, *saved)
            ctx.key, ctx.runtime, ctx.backend, ctx.plan, ctx.arena = key, runtime, backend, plan, arena
            ctx.home_shapes, ctx.need_scores = (home_gate.shape, home_down.shape), need_scores
        return buffers.output.to(hidden.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Preserve complete dW and source accumulation order while pipelining dMid/dX."""
        runtime, key, backend, plan = ctx.runtime, ctx.key, ctx.backend, ctx.plan
        executor, _ = runtime.executions[key]
        scores, source_output, full_x, full_a, *preacts = ctx.saved_tensors
        runtime.prepare(key)
        scheduler = getattr(executor.owner, 'hsdp_scheduler', None)
        if scheduler is None:
            runtime.prefetch_previous(key)
        streams, chunks, buffers = _prepare_phase(
            backend, plan, executor, grad_output, scores, True, ctx.need_scores,
            source_cache=source_output if ctx.need_scores else None)
        source = grad_output.reshape(-1, grad_output.shape[-1])
        full_dy = grad_output.new_empty((ctx.arena.rows, source.shape[-1]))
        full_du = grad_output.new_empty((ctx.arena.rows, full_a.shape[-1] * 2))
        buffers.direct_receive = ctx.arena.receive_views(full_dy, chunks)
        current_weights, home_gradients = [], []

        def weights() -> tuple:
            """Use the original version's rematerialized weights until the final dX."""
            if not current_weights:
                current_weights.extend(runtime.consume(key))
            return tuple(current_weights)

        def ordered(value: torch.Tensor) -> torch.Tensor:
            """Restore expert/source/token rows before the two full weight reductions."""
            if chunks.expert_restore is None:
                return value
            return value.index_select(0, chunks.expert_restore)

        def launch() -> None:
            """Publish both complete dW values before the last dX, using existing FSDP hooks."""
            with torch.profiler.record_function('moonep_chunked::weight_gradients'):
                streams.protect(chunks.expert_restore)
                down = grouped_weight_grad(ordered(full_a), ordered(full_dy), plan.boundaries)
                gate = grouped_weight_grad(ordered(full_x), ordered(full_du), plan.boundaries)
                if scheduler is None:
                    home_gradients.extend(return_home_gradients(gate, down, plan, executor.gradient_group))
                else:
                    home_gradients.extend(scheduler.launch_gradient_extension((gate, down, plan)))

        def pack(chunk: DeviceChunk, payload: torch.Tensor) -> None:
            """Fuse per-entry dY weighting and dScore reduction before this block's dispatch."""
            streams.protect(source, source_output, scores, chunk.source_tokens, chunk.source_choices,
                            chunk.send_indices, buffers.source_indices)
            selected = source.index_select(0, chunk.source_tokens)
            probabilities = scores.reshape(-1).index_select(0, chunk.source_choices).float()[:, None]
            if ctx.need_scores:
                indices = None if buffers.source_indices is None else buffers.source_indices[:selected.shape[0]]
                weighted, derivative = source_gradients(
                    selected, source_output[chunk.source_begin:chunk.source_end], probabilities, indices)
                payload.copy_(weighted)
                buffers.score_gradient.reshape(-1)[chunk.source_begin:chunk.source_end].copy_(derivative.reshape(-1))
            else:
                torch.mul(selected, probabilities, out=payload)

        def compute(ordinal: int, chunk: DeviceChunk, received: torch.Tensor) -> torch.Tensor:
            """Save row-local derivatives; each full dW sees the same rows as nonchunk execution."""
            gate, down = weights()
            begin, end = chunk.spec.weight_begin, chunk.spec.weight_end
            value = received.index_select(0, chunk.receive_order) if chunk.receive_order is not None else received
            value = ctx.arena.store(full_dy, ordinal, value)
            mid = grouped_values(value, down[begin:end].transpose(-1, -2), chunk.boundaries)
            du = activation_gradient(mid, preacts[ordinal], executor.limit, executor.activation)
            du = ctx.arena.store(full_du, ordinal, du)
            if ordinal + 1 == len(chunks):
                launch()
            return grouped_values(du, gate[begin:end].transpose(-1, -2), chunk.boundaries)

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
        if tuple(value.shape for value in home_gradients) != ctx.home_shapes:
            raise RuntimeError('chunked owner gradients differ from the real home parameter shapes')
        runtime.complete(key)
        ctx.plan, ctx.arena = None, None
        return (buffers.output.to(grad_output.dtype), score_gradient, *home_gradients,
                None, None, None, None, None)


def enable_chunked_moonep(model: torch.nn.Module, *, token_group: Any, chunk_rows: int = 4096) -> ChunkedMoonEP:
    """Install the Python chunk path on already-bound nonfused MoonEP experts.

    Args:
        model: Model with MoonEPExecutor owners; install before configure_moonep.
        token_group: Independent EP token group, created once in globally consistent order.
        chunk_rows: Maximum received entries per round, independent of EP degree.

    Returns:
        Shared backend owning token/merge streams and immutable per-call maps.

    Raises:
        ValueError: Missing nonfused experts, active executions, enabled dedup,
            or a token domain with different ordered EP members.
    """
    executors = [getattr(module, '_moonep_executor', None) for module in model.modules()]
    executors = [executor for executor in executors if isinstance(executor, MoonEPExecutor)]
    if not executors:
        raise ValueError('bind nonfused MoonEP experts before enabling the Python chunk pipeline')
    backend = ChunkedMoonEP(token_group, chunk_rows)
    for executor in executors:
        if executor.token_dedup:
            raise ValueError('the Python chunk transport currently requires moonep_token_dedup=False')
        if executor.runtime is not None:
            raise ValueError('install the chunk backend before configure_moonep')
        if executor.ranks > 1:
            if token_group is executor.group or token_group is None:
                raise ValueError('token communication needs an EP group independent of replica W/G communication')
            if dist.get_process_group_ranks(token_group) != dist.get_process_group_ranks(executor.group):
                raise ValueError('token group must preserve the ordered EP members')
    for executor in executors:
        executor.chunked = backend
    return backend
