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
"""Deduplicate token vectors by destination rank while retaining expert rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable

from hyper_parallel.distributed.expert_parallel.collectives import ep_all_to_all
from hyper_parallel.distributed.expert_parallel.moonep.device_planning import (
    HostControl, TableGraph, inverse_permutation,
)
from hyper_parallel.distributed.expert_parallel.moonep.peer_token_plan import (
    PeerTokenPlan, SourceGroupMap, _scatter_rows, layout_indices, source_tables, target_tables,
)


@dataclass(frozen=True)
class LocalDedup:
    """One rank's small metadata packet, prepared before collective admission."""

    packet: torch.Tensor
    packet_send: tuple[int, ...]
    packet_recv: tuple[int, ...]
    forward_tokens: torch.Tensor
    forward_send: torch.Tensor
    backward_tokens: torch.Tensor
    backward_send: torch.Tensor
    source_choices: torch.Tensor
    received_rows: int
    top_k: int = 1
    forward_members: torch.Tensor | None = None
    backward_members: torch.Tensor | None = None
    source_groups: SourceGroupMap | None = None
    valid: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    capabilities: int = 0


@dataclass(frozen=True)
class PacketRows:
    """Map vector rows and scalar lanes directly into the final opaque packet."""

    vectors: torch.Tensor
    scalar_lookup: torch.Tensor
    tails: torch.Tensor


@dataclass(frozen=True)
class TokenDedup:
    """Saved fanout and peer splits; forward rows may be compact or selection-dense."""

    forward_tokens: torch.Tensor
    forward_send: tuple[int, ...]
    forward_recv: tuple[int, ...]
    receive_lookup: torch.Tensor
    backward_tokens: torch.Tensor
    backward_send: tuple[int, ...]
    backward_recv: tuple[int, ...]
    backward_lookup: torch.Tensor
    source_choices: torch.Tensor
    width: int
    score_lanes: int
    group: Any
    target_hidden_rows: torch.Tensor
    target_score_elements: torch.Tensor
    source_split_rows: tuple[int, ...]
    target_split_rows: tuple[int, ...]
    source_packet: PacketRows
    backward_source_vector_rows: torch.Tensor
    backward_source_score_elements: torch.Tensor
    backward_source_split_rows: tuple[int, ...]
    backward_target_split_rows: tuple[int, ...]
    backward_packet: PacketRows
    forward_target_members: torch.Tensor
    forward_source_members: torch.Tensor
    backward_target_members: torch.Tensor
    backward_source_members: torch.Tensor
    reduction_scales: torch.Tensor
    peer_tokens: PeerTokenPlan
    complete_backward_inverse: torch.Tensor | None

    @property
    def saved_rows(self) -> int:
        """Count vector rows omitted from one forward transfer direction."""
        return self.source_choices.numel() - sum(self.forward_send)

    def _check_width(self, value: torch.Tensor) -> None:
        """Reject a route compiled for a different hidden-vector width."""
        if value.ndim != 2 or value.shape[-1] != self.width:
            raise ValueError("dedup vector width differs from the compiled peer layout")

    def _pack(self, vectors: torch.Tensor, scores: torch.Tensor, layout: PacketRows) -> torch.Tensor:
        """Write scalar tails into their final rows without a temporary vector concatenation."""
        if not layout.vectors.numel():
            return vectors.new_empty((0, self.width))
        packet = vectors.index_select(0, layout.vectors)
        lanes = torch.cat((scores.reshape(-1), scores.new_zeros(1))).index_select(0, layout.scalar_lookup)
        tails = lanes.reshape(-1, self.width)
        if packet.device.type == "npu":
            import torch_npu  # pylint: disable=import-outside-toplevel

            torch_npu.npu_scatter_nd_update_(packet, layout.tails.reshape(-1, 1), tails)
        else:
            packet.index_copy_(0, layout.tails.long(), tails)
        return packet

    def dispatch_with_scores(self, hidden: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack vectors and lossless BF16/FP32 score lanes by peer."""
        self._check_width(hidden)
        return _DispatchWithScores.apply(hidden, scores, self)

    def return_vectors(self, values: torch.Tensor, tokens: int) -> torch.Tensor:
        """Reduce expert rows locally before returning one vector per source token."""
        self._check_width(values)
        return _ReturnVectors.apply(values, tokens, self, None)

    def return_weighted(self, values: torch.Tensor, scores: torch.Tensor, tokens: int) -> torch.Tensor:
        """Combine expert values while retaining the router score precision.

        Args:
            values: Expert-major [selection, hidden] outputs.
            scores: One router score per expert-major selection.
            tokens: Number of original source tokens, including unrouted tokens.

        Returns:
            Source-token outputs. BF16 NPU values with FP32 scores use CANN
            combine and its backward; other dtypes retain the portable path.
        """
        if values.device.type != "npu" or values.dtype != torch.bfloat16 or scores.dtype != torch.float32:
            return self.return_vectors(values * scores[:, None], tokens)
        self._check_width(values)
        return _ReturnVectors.apply(values, tokens, self, scores)

    def sum_rows(self, values: torch.Tensor, lookup: torch.Tensor, rows: int,
                 members: torch.Tensor) -> torch.Tensor:
        """Sum row groups without the generic deterministic FP32 scatter kernel.

        Args:
            values: Selection or partial vectors, retaining their numeric dtype.
            lookup: Destination row per input, used by the portable implementation.
            rows: Number of output groups, including empty source token rows.
            members: Row-major INT32 member table; -1 denotes an absent member.

        Returns:
            Group sums with the original dtype. BF16 retains its existing
            scatter/write rounding; FP32 NPU sums use CANN's deterministic
            gather-and-reduce operation, with one writer per output vector.
        """
        if values.device.type == "npu" and values.dtype == torch.float32 and rows and values.shape[0]:
            # TorchNPU is optional for CPU/CUDA users. Drop mode admits ragged
            # members; the singleton capacity axis is a view, not H-vector padding.
            import torch_npu  # pylint: disable=import-outside-toplevel

            table = members.reshape(rows, -1)
            fan_in = min(table.shape[1], values.shape[0])
            return torch_npu.npu_moe_finalize_routing(
                values.unsqueeze(1), None, None, None, self.reduction_scales[:rows, :fan_in],
                table[:, :fan_in].reshape(-1), None, 3)
        result = values.new_zeros((rows, self.width))
        return result.index_add_(0, lookup, values)


class _DispatchWithScores(torch.autograd.Function):
    """Use the saved forward map while choosing an independent reverse layout."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, scores: torch.Tensor,
                plan: TokenDedup) -> tuple[torch.Tensor, torch.Tensor]:
        """Send the selected forward vector layout and every expert's router score."""
        ctx.plan, ctx.hidden_shape, ctx.score_shape, ctx.score_dtype = plan, hidden.shape, scores.shape, scores.dtype
        vectors = hidden.index_select(0, plan.forward_tokens)
        if scores.dtype not in (torch.bfloat16, torch.float32) or max(
                1, scores.element_size() // hidden.element_size()) != plan.score_lanes:
            raise ValueError("dedup score dtype differs from the compiled peer layout")
        ctx.transport_score_dtype = hidden.dtype if scores.element_size() < hidden.element_size() else scores.dtype
        selected_scores = scores.reshape(-1).index_select(0, plan.source_choices).contiguous()
        selected_scores = (selected_scores.to(hidden.dtype) if scores.element_size() < hidden.element_size()
                           else selected_scores.view(hidden.dtype))
        payload = plan._pack(vectors, selected_scores, plan.source_packet)
        received = ep_all_to_all(payload, list(plan.source_split_rows), list(plan.target_split_rows), plan.group)
        return (received.index_select(0, plan.target_hidden_rows),
                received.reshape(-1).index_select(
                    0, plan.target_score_elements).contiguous().view(ctx.transport_score_dtype))

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_hidden: torch.Tensor, grad_scores: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Return dX using the configured reverse peer layout and preserve every dScore."""
        plan = ctx.plan
        compact = plan.sum_rows(grad_hidden, plan.backward_lookup, sum(plan.backward_recv),
                                plan.backward_target_members)
        score_lanes = grad_scores.reshape(-1).contiguous().view(grad_hidden.dtype)
        payload = plan._pack(compact, score_lanes, plan.backward_packet)
        returned = ep_all_to_all(payload, list(plan.backward_target_split_rows),
                                 list(plan.backward_source_split_rows), plan.group)
        vector_gradients = returned.index_select(0, plan.backward_source_vector_rows)
        if plan.width % plan.score_lanes == 0:
            score_gradients = returned.reshape(-1).view(ctx.transport_score_dtype).index_select(
                0, plan.backward_source_score_elements)
        else:
            score_gradients = returned.reshape(-1).index_select(
                0, plan.backward_source_score_elements).contiguous().view(ctx.transport_score_dtype)
        hidden_gradient = plan.sum_rows(vector_gradients, plan.backward_tokens, ctx.hidden_shape[0],
                                        plan.backward_source_members)
        return hidden_gradient, score_gradients.to(ctx.score_dtype).reshape(ctx.score_shape), None


class _ReturnVectors(torch.autograd.Function):
    """Reduce at the expert owner and reuse independent reverse-route metadata."""

    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, tokens: int, plan: TokenDedup,
                scores: torch.Tensor | None) -> torch.Tensor:
        """Optionally combine FP32 scores without materializing a weighted hidden table."""
        ctx.plan, ctx.values_shape, ctx.weighted = plan, values.shape, scores is not None
        rows = sum(plan.forward_recv)
        if scores is None:
            compact = plan.sum_rows(values, plan.receive_lookup, rows, plan.forward_target_members)
        else:
            import torch_npu  # pylint: disable=import-outside-toplevel

            members = plan.forward_target_members.reshape(rows, -1) if rows else None
            if rows:
                scales = scores.reshape(-1).index_select(0, members.clamp_min(0).long().reshape(-1)).reshape_as(members)
                scales = scales.masked_fill(members < 0, 0)
                compact = torch_npu.npu_moe_finalize_routing(
                    values.float().unsqueeze(1), None, None, None, scales, members.reshape(-1), None, 3)
            else:
                compact = torch.empty((0, plan.width), dtype=torch.float32, device=values.device)
            identity = (torch.empty(0, dtype=torch.int32, device=values.device)
                        if plan.complete_backward_inverse is not None else
                        torch.arange(values.shape[0], dtype=torch.int32, device=values.device))
            ctx.save_for_backward(values, scores, identity)
        returned = ep_all_to_all(compact, list(plan.forward_recv), list(plan.forward_send), plan.group)
        result = plan.sum_rows(returned, plan.forward_tokens, tokens, plan.forward_source_members)
        return result.to(values.dtype) if scores is not None else result

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Send compact G and use CANN reverse combine for dV and independent dScore."""
        plan = ctx.plan
        payload = grad_output.index_select(0, plan.backward_tokens)
        received = ep_all_to_all(payload, list(plan.backward_send), list(plan.backward_recv), plan.group)
        if not ctx.weighted:
            return received.index_select(0, plan.backward_lookup).reshape(ctx.values_shape), None, None, None
        values, scores, identity = ctx.saved_tensors
        if plan.complete_backward_inverse is not None:
            members = plan.backward_target_members
            weights = scores.reshape(-1).index_select(0, members).reshape(-1, plan.peer_tokens.source_shape[1])
            dv, grouped_scores = torch.ops.npu.npu_moe_token_unpermute_grad(values, received, members, weights)
            ds = grouped_scores.reshape(-1).index_select(0, plan.complete_backward_inverse)
        elif values.shape[0]:
            expanded = received.index_select(0, plan.backward_lookup)
            dv, ds = torch.ops.npu.npu_moe_token_unpermute_grad(values, expanded, identity, scores.reshape(-1, 1))
        else:
            dv, ds = torch.empty_like(values), torch.empty_like(scores)
        return dv, None, None, ds.reshape_as(scores)


class _PlanTables:
    """Own retained maps through one backing allocation per metadata dtype."""

    def __init__(self, function: Callable, example: torch.Tensor) -> None:
        """Group graph outputs by dtype while keeping shape and offset descriptions."""
        self.function = function
        self.layout = []
        self.graph = TableGraph(self._packed, example)

    def _packed(self, *arguments: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Fuse packing in the graph so ownership needs no per-map clone launch."""
        outputs = self.function(*arguments)
        dtypes = tuple(dict.fromkeys(value.dtype for value in outputs))
        offsets = [0] * len(dtypes)
        self.layout = []
        for value in outputs:
            group = dtypes.index(value.dtype)
            self.layout.append((value.shape, group, offsets[group], value.numel()))
            offsets[group] += value.numel()
        return tuple(torch.cat(tuple(value.reshape(-1) for value in outputs if value.dtype == dtype))
                     for dtype in dtypes)

    def __call__(self, *arguments: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return this invocation owned output views with their original tensor shapes."""
        buffers = self.graph(*arguments)
        return tuple(buffers[group].narrow(0, offset, size).reshape(shape)
                     for shape, group, offset, size in self.layout)


def _table(graphs: dict, key: tuple, function: Callable, *arguments: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Reuse one topology program; each invocation retains its own tensor outputs."""
    if arguments[0].device.type != 'npu':
        return function(*arguments)
    if key not in graphs:
        graphs[key] = _PlanTables(function, arguments[0])
    return graphs[key](*arguments)


def prepare_local_dedup(order: torch.Tensor, token_send: Sequence[int], token_recv: Sequence[int],
                        top_k: int, *, forward_ratio: float = 0.75, backward_ratio: float = 1.0,
                        graphs: dict | None = None, capabilities: int = 0,
                        keep_groups: bool = False) -> LocalDedup:
    """Prepare fixed-capacity device groups and a fixed-length metadata packet.

    Args:
        order: Final target-major original Top-K choices; its producer owns the permutation.
        token_send: Base selection counts sent to each target.
        token_recv: Base selection counts received from each source.
        top_k: Original router fanout, also the maximum group membership.
        forward_ratio: Inclusive forward unique-fraction threshold.
        backward_ratio: Strict backward unique-fraction threshold.
        graphs: Model-owned cache of small topology programs.
        capabilities: Protocol/numeric policy agreed before any payload.
        keep_groups: Static native-consumer capability; not a routing decision.

    Returns:
        An unfinished device plan. Its token maps retain fixed source capacity;
        send-count tensors are not read by the host until finish_dedup.
    """
    ranks, rows = len(token_send), order.numel()
    if (ranks < 1 or len(token_recv) != ranks or top_k < 1
            or not 0.0 <= forward_ratio <= 1.0 or not 0.0 <= backward_ratio <= 1.0
            or any(count < 0 for count in (*token_send, *token_recv))
            or order.ndim != 1 or order.dtype != torch.int64 or sum(token_send) != rows
            or isinstance(capabilities, bool) or not isinstance(capabilities, int) or not 0 <= capabilities < 32768
            or rows % top_k or max(rows, sum(token_recv)) * top_k >= 2**31
            or not isinstance(keep_groups, bool)):
        raise ValueError("dedup source geometry exceeds the fixed member-table coordinates")
    graphs = {} if graphs is None else graphs
    controls = torch.tensor([
        list(token_send), list(token_recv),
        [math.floor(forward_ratio * count) for count in token_send],
        [math.ceil(backward_ratio * count) - 1 for count in token_send]],
        dtype=torch.int32, device='cpu')
    if order.device.type != 'cpu':
        controls = controls.pin_memory(device=order.device).to(order.device, non_blocking=True)

    def program(choices: torch.Tensor, dynamic: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Build device source maps from runtime counts and configured ratio limits."""
        limits = torch.stack((dynamic[0], dynamic[2], dynamic[3]))
        return source_tables(choices, limits, top_k, capabilities, keep_groups=keep_groups)

    key = ('dedup_source', str(order.device), rows, ranks, top_k, capabilities, keep_groups)
    outputs = _table(graphs, key, program, order, controls)
    (packet, forward_tokens, backward_tokens, forward_send, backward_send,
     forward_members, backward_members) = outputs[:7]
    groups = SourceGroupMap(*outputs[7:-1]) if keep_groups else None
    return LocalDedup(
        packet=packet, packet_send=tuple(3 * (count + 1) for count in token_send),
        packet_recv=tuple(3 * (count + 1) for count in token_recv),
        forward_tokens=forward_tokens, forward_send=forward_send,
        backward_tokens=backward_tokens, backward_send=backward_send,
        source_choices=order.detach(), received_rows=sum(token_recv), top_k=top_k,
        forward_members=forward_members, backward_members=backward_members,
        source_groups=groups, valid=outputs[-1], counts=controls, capabilities=capabilities)


def failed_local_dedup(token_send: Sequence[int], token_recv: Sequence[int], *,
                       device: torch.device | None = None, top_k: int = 1) -> LocalDedup:
    """Broadcast poison headers through the same metadata collective after a local failure."""
    device = torch.device('cpu') if device is None else device
    ranks, rows = len(token_send), sum(token_send)
    counts = torch.tensor([list(token_send), list(token_recv), [0] * ranks, [-1] * ranks],
                          dtype=torch.int32, device=device)
    packet = torch.zeros(3 * (rows + ranks), dtype=torch.int32, device=device)
    offsets = ((counts[0] + 1).cumsum(0, dtype=torch.int32) - counts[0] - 1) * 3
    packet.index_fill_(0, offsets.long(), -1)
    tokens = torch.empty(0, dtype=torch.int32, device=device)
    return LocalDedup(
        packet, tuple(3 * (count + 1) for count in token_send), tuple(3 * (count + 1) for count in token_recv),
        tokens, torch.zeros(ranks, dtype=torch.int32, device=device), tokens,
        torch.zeros(ranks, dtype=torch.int32, device=device), torch.zeros(rows, dtype=torch.int64, device=device),
        sum(token_recv), top_k, valid=torch.zeros((), dtype=torch.bool, device=device), counts=counts)


def _packet_rows(order: torch.Tensor, positions: torch.Tensor, vectors: torch.Tensor,
                 capacity: int, width: int) -> tuple[torch.Tensor, ...]:
    """Build final-packet row ownership without making dynamic counts a graph key."""
    rows, lanes = order.numel(), positions.numel()
    if max(2 * rows, lanes) > torch.iinfo(torch.int32).max:
        raise ValueError("packet row metadata exceeds INT32 index capacity")
    if capacity == 0:
        empty = order.new_empty(0)
        return empty, empty, empty
    count = vectors.sum(dtype=torch.int32)
    tails = (positions.max() + width) // width if lanes else count.new_zeros(())
    ids = torch.arange(rows, dtype=torch.int32, device=order.device)
    inverse = _scatter_rows(ids.new_zeros(2 * rows),
                            torch.where(ids < count + tails, order.int(), rows + ids), ids)[:rows]
    tail_ids = torch.arange(rows - capacity, dtype=torch.int32, device=order.device) + count
    tail_rows = inverse.index_select(0, tail_ids.clamp(0, rows - 1))
    lookup = torch.full(((rows - capacity) * width,), lanes, dtype=torch.int32, device=order.device)
    lookup = _scatter_rows(lookup, positions, torch.arange(lanes, dtype=torch.int32, device=order.device))
    return torch.minimum(order.int().clamp_min(0), (count - 1).clamp_min(0)), lookup, tail_rows


def finish_dedup(local: LocalDedup, group: Any, *, device: torch.device, width: int,
                 backward_ratio: float = 1.0, score_lanes: int = 1,
                 graphs: dict | None = None, execution: Any = None,
                 source_inverse: torch.Tensor | None = None,
                 target_order: torch.Tensor | None = None) -> TokenDedup:
    """Exchange device descriptors and export one small, globally admitted split packet.

    Args:
        local: Fixed-capacity source groups, or poison headers after preparation failure.
        group: Existing EP group, reused for metadata and status convergence.
        device: Expected metadata device; cross-device source plans are rejected.
        width: Hidden-vector width in payload elements.
        backward_ratio: Retained call contract; the source already encoded its decision.
        score_lanes: One or two transport elements per original score.
        graphs: Model-owned topology program cache.
        execution: Invocation identity shared by original forward, replay and backward.
        source_inverse: Optional producer-validated inverse of source_choices.
            The base route already owns this map; other callers build it once on device.
        target_order: Optional producer-validated permutation from wire rows to expert rows.

    Returns:
        Compact data-plane maps and host splits. Only O(R) controls cross D2H;
        original choices, group members and packet permutations stay on device.
    """
    if (not 0.0 <= backward_ratio <= 1.0 or score_lanes not in (1, 2) or width < 1
            or local.packet.device != device):
        raise ValueError("dedup packet dtype, width or device differs from its source plan")
    graphs = {} if graphs is None else graphs
    ranks = len(local.packet_send)
    received = torch.empty(sum(local.packet_recv), dtype=torch.int32, device=device)
    if ranks == 1:
        received.copy_(local.packet)
    else:
        dist.all_to_all_single(received, local.packet, output_split_sizes=list(local.packet_recv),
                               input_split_sizes=list(local.packet_send), group=group)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    error = None
    target = layouts = scales = None
    try:
        if local.valid is None or local.counts is None:
            raise RuntimeError("dedup source preparation failed")
        receive_rows, source_rows, top_k = local.received_rows, local.source_choices.numel(), local.top_k
        capabilities = local.capabilities
        if target_order is not None and (target_order.device != device or target_order.shape != (receive_rows,)
                                          or target_order.dtype not in (torch.int32, torch.int64)):
            raise ValueError("target order must match this invocation's expert-row permutation")
        expert_major = target_order is not None
        target_permutation = torch.empty(0, dtype=torch.int64, device=device) if not expert_major else target_order
        if source_inverse is not None and (
                source_inverse.device != device or source_inverse.shape != local.source_choices.shape
                or source_inverse.dtype not in (torch.int32, torch.int64)):
            raise ValueError("source inverse must match this invocation's source-choice permutation")
        inverse = (torch.empty(0, dtype=torch.int32, device=device)
                   if source_inverse is None else source_inverse)

        def receive_program(packet: torch.Tensor, controls: torch.Tensor,
                            order: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Decode peer metadata directly into the consumer expert-row coordinate system."""
            row_values = inverse_permutation(order.int()) if expert_major else None
            result = target_tables(packet, controls[1], receive_rows, top_k, capabilities, row_values=row_values)
            if capabilities & 32:
                # Native packets consume member maps directly, not opaque-frame lookups.
                empty = packet.new_empty(0)
                return result[:2] + (empty, empty) + result[4:]
            if not expert_major or receive_rows == 0:
                return result
            return (result[0], result[1], result[2].index_select(0, order), result[3].index_select(0, order),
                    result[4], result[5], result[6])

        target_key = ('dedup_target', str(device), source_rows, receive_rows, ranks, top_k,
                      capabilities, target_order is not None)
        target = _table(graphs, target_key, receive_program, received, local.counts, target_permutation)
        forward_recv, backward_recv, lookup, backward_lookup = target[:4]

        def packet_program(fs: torch.Tensor, bs: torch.Tensor, fr: torch.Tensor, br: torch.Tensor,
                           counts: torch.Tensor, expand: torch.Tensor, choices: torch.Tensor,
                           inverse: torch.Tensor, order: torch.Tensor,
                           backward_members: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Compose all four packet layouts using fixed admission capacities."""
            source_scores, target_scores = counts[0] * score_lanes, counts[1] * score_lanes
            # Only selected outputs survive graph compilation; all extents are fixed capacities.
            source_fd = layout_indices(fs, source_scores, width, source_rows, source_rows * score_lanes)
            target_fd = layout_indices(fr, target_scores, width, receive_rows, receive_rows * score_lanes)
            source_bc = layout_indices(bs, source_scores, width, source_rows, source_rows * score_lanes)
            target_bc = layout_indices(br, target_scores, width, receive_rows, receive_rows * score_lanes)
            if source_rows:
                if inverse.numel() == 0:
                    inverse = inverse_permutation(choices.int().clamp(0, source_rows - 1))
                # Invalid source metadata must reach the common status boundary
                # instead of causing an out-of-bounds gather before convergence.
                inverse = inverse.clamp(0, source_rows - 1)
            source_score_elements = source_bc[1].reshape(source_rows, score_lanes)
            if width % score_lanes == 0:
                source_score_elements = source_score_elements[:, 0] // score_lanes
            source_score_elements = source_score_elements.index_select(0, inverse).reshape(-1)
            hidden_rows = target_fd[0].index_select(0, expand)
            score_elements, score_positions = target_fd[1], target_bc[3]
            if expert_major and receive_rows:
                score_elements = score_elements.reshape(receive_rows, score_lanes).index_select(0, order).reshape(-1)
                score_positions = score_positions.reshape(receive_rows, score_lanes).index_select(0, order).reshape(-1)
            source_packet = _packet_rows(source_fd[2], source_fd[3], fs, source_rows, width)
            backward_packet = _packet_rows(target_bc[2], score_positions, br, receive_rows, width)
            complete_inverse = torch.empty(0, dtype=torch.int32, device=device)
            if receive_rows and score_lanes == 2:
                choices = backward_members.reshape(-1)[:receive_rows]
                positions = torch.arange(receive_rows, dtype=torch.int32, device=device)
                complete_inverse = _scatter_rows(positions.new_zeros(2 * receive_rows),
                    torch.where(choices >= 0, choices, receive_rows + positions), positions)[:receive_rows]
            return (hidden_rows, score_elements, *source_packet, source_bc[0], source_score_elements,
                    *backward_packet, complete_inverse)

        key = ('dedup_packets', str(device), source_rows, receive_rows, ranks, width, score_lanes,
               source_inverse is not None, inverse.dtype, target_order is not None)
        if capabilities & 32:
            layouts = (local.source_choices.new_empty(0),) * 11
            scales = torch.ones((source_rows // top_k, top_k), dtype=torch.float32, device=device)
        else:
            layouts = _table(graphs, key, packet_program, local.forward_send, local.backward_send,
                             forward_recv, backward_recv, local.counts, lookup, local.source_choices, inverse,
                             target_permutation, target[5])
            scales = torch.ones((max(receive_rows, source_rows // top_k), top_k), dtype=torch.float32, device=device)
        status.copy_((~(local.valid & target[-1])).int().reshape(1))
    except Exception as exc:
        error = exc
        status.fill_(1)
    if ranks > 1:
        # This one planning-time status collective covers target compilation and
        # allocation failures too. Replay and backward submit no such collective.
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=group)
    zeros = local.forward_send.new_zeros(ranks)
    control = torch.cat((status, local.forward_send, zeros if target is None else target[0],
                          local.backward_send, zeros if target is None else target[1])).long()
    header = HostControl(control).tolist()
    if header[0]:
        raise RuntimeError("dedup metadata or preparation failed on an EP rank") from error
    forward_send, forward_recv, backward_send, backward_recv = (
        tuple(header[1 + index * ranks:1 + (index + 1) * ranks]) for index in range(4))
    source_rows = tuple(count // 3 - 1 for count in local.packet_send)
    target_rows = tuple(count // 3 - 1 for count in local.packet_recv)

    def packet_splits(vectors: Sequence[int], selections: Sequence[int]) -> tuple[int, ...]:
        """Add each peer scalar-tail rows to its active hidden-vector row count."""
        return tuple(vector + (count * score_lanes + width - 1) // width
                     for vector, count in zip(vectors, selections))

    source_splits = packet_splits(forward_send, source_rows)
    target_splits = packet_splits(forward_recv, target_rows)
    backward_source_splits = packet_splits(backward_send, source_rows)
    backward_target_splits = packet_splits(backward_recv, target_rows)
    (hidden_rows, score_elements, source_vectors, source_scalars, source_tails, bc_rows, bc_scores,
     target_vectors, target_scalars, target_tails, complete_inverse) = layouts
    source_tail_rows = sum(source_splits) - sum(forward_send)
    target_tail_rows = sum(backward_target_splits) - sum(backward_recv)
    peer_tokens = PeerTokenPlan(execution, (local.source_choices.numel() // local.top_k, local.top_k))
    return TokenDedup(
        forward_tokens=local.forward_tokens[:sum(forward_send)], forward_send=forward_send, forward_recv=forward_recv,
        receive_lookup=target[2], backward_tokens=local.backward_tokens[:sum(backward_send)],
        backward_send=backward_send, backward_recv=backward_recv, backward_lookup=target[3],
        source_choices=local.source_choices, width=width, score_lanes=score_lanes, group=group,
        target_hidden_rows=hidden_rows, target_score_elements=score_elements,
        source_split_rows=source_splits, target_split_rows=target_splits,
        source_packet=PacketRows(source_vectors[:sum(source_splits)], source_scalars[:source_tail_rows * width],
                                 source_tails[:source_tail_rows]),
        backward_source_vector_rows=bc_rows[:sum(backward_send)], backward_source_score_elements=bc_scores,
        backward_source_split_rows=backward_source_splits, backward_target_split_rows=backward_target_splits,
        backward_packet=PacketRows(target_vectors[:sum(backward_target_splits)],
                                   target_scalars[:target_tail_rows * width], target_tails[:target_tail_rows]),
        forward_target_members=target[4][:sum(forward_recv)].reshape(-1),
        forward_source_members=local.forward_members.reshape(-1),
        backward_target_members=target[5][:sum(backward_recv)].reshape(-1),
        backward_source_members=local.backward_members.reshape(-1), reduction_scales=scales, peer_tokens=peer_tokens,
        complete_backward_inverse=(complete_inverse if complete_inverse.numel() and score_lanes == 2
                                   and receive_rows == sum(backward_recv) * top_k else None))
