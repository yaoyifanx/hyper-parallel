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
"""Compile receiver-bounded Python chunks from MoonEP's existing CPU histogram."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import Iterator

import numpy as np
import torch

from hyper_parallel.distributed.expert_parallel.moonep.planning import ExecutionPlan


@dataclass(frozen=True)
class ChunkSpec:
    """One globally ordered round; local receive rows are bounded by chunk_rows."""

    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    send_ranges: tuple[tuple[int, int], ...]
    receive_ranges: tuple[tuple[int, int], ...]
    boundaries: tuple[int, ...]
    weight_begin: int
    weight_end: int


@dataclass(frozen=True)
class DeviceChunk:
    """Immutable indices reused by original forward, replay and backward."""

    spec: ChunkSpec
    send_indices: torch.Tensor
    source_choices: torch.Tensor
    source_tokens: torch.Tensor
    receive_order: torch.Tensor | None
    restore_order: torch.Tensor | None
    boundaries: torch.Tensor
    source_begin: int
    source_end: int


@dataclass(frozen=True)
class ChunkLayout:
    """Per-execution chunks and the original-choice to contiguous-cache permutation."""

    chunks: tuple[DeviceChunk, ...]
    source_order: torch.Tensor
    expert_restore: torch.Tensor | None = None
    source_gradient_order: torch.Tensor | None = None

    def __iter__(self) -> Iterator[DeviceChunk]:
        """Visit chunks in their shared communication order."""
        return iter(self.chunks)

    def __len__(self) -> int:
        """Return the common number of communication rounds."""
        return len(self.chunks)

    def __getitem__(self, index: int) -> DeviceChunk:
        """Return one already compiled communication round."""
        return self.chunks[index]


def _ranges(first: np.ndarray, rows: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(start), int(count)) for start, count in zip(first, rows) if count)


def _peer_round_quotas(source: np.ndarray, target: np.ndarray, rounds: int) -> np.ndarray:
    """Divide actual peer traffic across rounds with balanced receiver capacity.

    Every peer contributes its quotient in every round. Remainder rows cycle
    across rounds, so receiver loads differ by at most one row and sender loads
    differ by at most the peer count. The remainder phase minimizes the added
    squared sender load; its search uses circular prefix sums, not a solver.
    """
    ranks, experts = source.shape
    traffic = np.zeros((ranks, ranks), dtype=np.int64)
    for expert in range(experts):
        senders = np.flatnonzero(source[:, expert])
        receivers = np.flatnonzero(target[expert])
        left, right = 0, 0
        sent, received = 0, 0
        while left < len(senders) and right < len(receivers):
            sender, receiver = senders[left], receivers[right]
            count = min(source[sender, expert] - sent, target[expert, receiver] - received)
            traffic[sender, receiver] += count
            sent, received = sent + count, received + count
            if sent == source[sender, expert]:
                left, sent = left + 1, 0
            if received == target[expert, receiver]:
                right, received = right + 1, 0
    if rounds == 0:
        return np.empty((0, ranks, ranks), dtype=np.int64)
    quotas = np.broadcast_to(traffic // rounds, (rounds, ranks, ranks)).copy()
    remainder = traffic % rounds
    loads = np.zeros((ranks, rounds), dtype=np.int64)
    for receiver in np.argsort(-remainder.sum(axis=0), kind='stable'):
        counts = remainder[:, receiver]
        starts = counts.cumsum() - counts
        costs = np.zeros(rounds, dtype=np.int64)
        for sender in np.flatnonzero(counts):
            count = int(counts[sender])
            prefix = np.concatenate(([0], np.tile(loads[sender], 2).cumsum()))
            windows = prefix[count:count + rounds] - prefix[:rounds]
            costs += np.roll(windows, -int(starts[sender] % rounds))
        phase = int(costs.argmin())
        for sender in np.flatnonzero(counts):
            ordinals = (starts[sender] + phase + np.arange(counts[sender])) % rounds
            quotas[ordinals, sender, receiver] += 1
            loads[sender, ordinals] += 1
    return quotas


def chunk_specs(plan: ExecutionPlan, rank: int, chunk_rows: int) -> tuple[ChunkSpec, ...]:
    """Split actual peer rows before regrouping each receiver's experts.

    Args:
        plan: Existing placement with its CPU source_counts histogram retained.
        rank: Rank in the original EP group.
        chunk_rows: Positive maximum received entries per local compute round.

    Returns:
        Globally matching peer splits, balanced across the same number of
        rounds on every rank. Receive ranges restore expert-major computation.
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    if plan.source_counts is None or plan.source_counts.device.type != "cpu":
        raise ValueError("chunk planning requires the original CPU MoonEP source histogram")
    source, target = plan.source_counts.numpy(), plan.allocation.numpy()
    ranks, experts = source.shape
    if target.shape != (experts, ranks) or not 0 <= rank < ranks:
        raise ValueError("chunk placement and EP rank do not match the source histogram")
    if np.any(source < 0) or np.any(target < 0) or not np.array_equal(source.sum(0), target.sum(1)):
        raise ValueError("chunk source and target expert counts must be nonnegative and conserved")
    source_end, target_end = source.cumsum(0), target.cumsum(1)
    source_begin, target_begin = source_end - source, target_end - target
    send_first = np.maximum(source_begin[rank, :, None], target_begin)
    send_rows = np.maximum(np.minimum(source_end[rank, :, None], target_end) - send_first, 0)
    receive_first = np.maximum(source_begin, target_begin[None, :, rank])
    receive_rows = np.maximum(np.minimum(source_end, target_end[None, :, rank]) - receive_first, 0)
    send_positions = send_rows.cumsum(0) - send_rows
    receive_positions = receive_rows.cumsum(1) - receive_rows
    source_peers = np.asarray([0, *accumulate(plan.token_send)], dtype=np.int64)[:-1]
    active = np.flatnonzero(target[:, rank])
    rounds = (int(target.sum(0).max()) + chunk_rows - 1) // chunk_rows
    quotas = _peer_round_quotas(source, target, rounds)
    sent = np.zeros(ranks, dtype=np.int64)
    received = np.zeros(ranks, dtype=np.int64)
    result = []
    for quota in quotas:
        send_end, receive_end = sent + quota[rank], received + quota[:, rank]
        outgoing_first = np.maximum(send_positions, sent[None])
        outgoing = np.maximum(np.minimum(send_positions + send_rows, send_end[None]) - outgoing_first, 0)
        send_starts = source_peers[None] + outgoing_first
        incoming_first = np.maximum(receive_positions, received[:, None])
        incoming = np.maximum(np.minimum(receive_positions + receive_rows, receive_end[:, None]) - incoming_first, 0)
        receive_starts = incoming.reshape(-1).cumsum() - incoming.reshape(-1)
        # The collective receives peer-major rows; GEMMs consume expert-major rows.
        receive_ranges = _ranges(receive_starts.reshape(ranks, experts).T.reshape(-1), incoming.T.reshape(-1))
        counts = incoming.sum(0)[active]
        used = np.flatnonzero(counts)
        low, high = (int(used[0]), int(used[-1]) + 1) if used.size else (0, 0)
        if int(incoming.sum()) > chunk_rows:
            raise ValueError("chunk receiver quota exceeds its compute-row capacity")
        result.append(ChunkSpec(tuple(map(int, quota[rank])), tuple(map(int, quota[:, rank])),
                                _ranges(send_starts.T.reshape(-1), outgoing.T.reshape(-1)),
                                receive_ranges, tuple(map(int, counts[low:high].cumsum())), low, high))
        sent, received = send_end, receive_end
    return tuple(result)


def _expand(ranges: tuple[tuple[int, int], ...], device: torch.device) -> torch.Tensor:
    starts, counts = zip(*ranges) if ranges else ((), ())
    total = sum(counts)
    if not ranges:
        return torch.empty(0, dtype=torch.int64, device=device)
    if all(start == starts[0] + previous for start, previous in zip(starts, [0, *accumulate(counts)])):
        return torch.arange(starts[0], starts[0] + total, device=device)
    sizes = np.asarray(counts, dtype=np.int64)
    offsets = np.asarray(starts, dtype=np.int64) - (sizes.cumsum() - sizes)
    expanded = torch.from_numpy(np.repeat(offsets, sizes) + np.arange(total, dtype=np.int64))
    if device.type == 'cpu':
        return expanded
    # Ranges already reside on the CPU; do not spend device kernels expanding these small index tables.
    return expanded.pin_memory(device=device).to(device, non_blocking=True)


def _receive_maps(spec: ChunkSpec, device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    offsets = list(accumulate((count for _, count in spec.receive_ranges), initial=0))
    if all(start == offset for (start, _), offset in zip(spec.receive_ranges, offsets)):
        return None, None
    inverse = tuple((offset, count) for _, offset, count in sorted(
        (start, offset, count) for (start, count), offset in zip(spec.receive_ranges, offsets)))
    return _expand(spec.receive_ranges, device), _expand(inverse, device)


def _upload(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Upload CPU metadata asynchronously; its consumer already waits on the caller stream."""
    if device.type == 'cpu':
        return value
    return value.pin_memory(device=device).to(device, non_blocking=True)


def _expert_restore(plan: ExecutionPlan, chunks: tuple[DeviceChunk, ...], rank: int) -> np.ndarray:
    """Map the chunk expert arena to the nonchunk reduction order.

    No device contents, new collective, or data-dependent device count is read.
    The returned permutation indexes the existing arena in canonical expert,
    source-rank, and token order before the two complete weight-gradient GMMs.
    """
    source, target = plan.source_counts.numpy(), plan.allocation.numpy()
    source_end, target_end = source.cumsum(0), target.cumsum(1)
    counts = np.maximum(np.minimum(source_end, target_end[None, :, rank])
                        - np.maximum(source_end - source, (target_end-target)[None, :, rank]), 0)
    active = np.flatnonzero(target[:, rank])
    expert_sizes = counts.sum(0)
    expert_bases = expert_sizes.cumsum() - expert_sizes
    peer_sizes = counts.sum(1)
    peer_bases = peer_sizes.cumsum() - peer_sizes
    total = int(peer_sizes.sum())
    wire_to_canonical = np.empty(total, dtype=np.int64)
    used_expert = np.zeros(counts.shape[1], dtype=np.int64)
    for peer in range(len(peer_sizes)):
        cursor = int(peer_bases[peer])
        for expert in active:
            size = int(counts[peer, expert])
            first = int(expert_bases[expert] + used_expert[expert])
            wire_to_canonical[cursor:cursor+size] = np.arange(first, first+size)
            cursor += size
            used_expert[expert] += size
    append_to_canonical = np.full(total, -1, dtype=np.int64)
    local_bases = list(accumulate(plan.group_counts, initial=0))
    used_local, consumed = [0] * len(plan.group_counts), np.zeros(len(peer_sizes), dtype=np.int64)
    for part in chunks:
        pieces = [np.arange(int(peer_bases[peer] + consumed[peer]),
                            int(peer_bases[peer] + consumed[peer] + size))
                  for peer, size in enumerate(part.spec.receive_counts)]
        wire = np.concatenate(pieces)
        order = (np.concatenate([np.arange(begin, begin + size) for begin, size in part.spec.receive_ranges])
                 if part.spec.receive_ranges else np.empty(0, dtype=np.int64))
        canonical = wire_to_canonical[wire[order]]
        previous = 0
        for local, end in enumerate(part.spec.boundaries):
            expert = part.spec.weight_begin + local
            size = end - previous
            first = local_bases[expert] + used_local[expert]
            append_to_canonical[first:first+size] = canonical[previous:end]
            used_local[expert] += size
            previous = end
        consumed += part.spec.receive_counts
    if tuple(used_local) != plan.group_counts or not np.array_equal(np.sort(append_to_canonical), np.arange(total)):
        raise ValueError('canonical restoration does not cover every expert row exactly once')
    return np.argsort(append_to_canonical, kind='stable')


def compile_chunks(plan: ExecutionPlan, rank: int, chunk_rows: int, top_k: int) -> ChunkLayout:
    """Upload only segment metadata; leave all original token choices on the device.

    Args:
        plan: Original immutable MoonEP placement and dispatch permutation.
        rank: EP-local source/target rank.
        chunk_rows: Maximum local compute rows per round.
        top_k: Router choices per source token.

    Returns:
        Device maps, exact grouped-GEMM boundaries and one cached inverse source map.
    """
    chunks, source_begin = [], 0
    for spec in chunk_specs(plan, rank, chunk_rows):
        send = _expand(spec.send_ranges, plan.dispatch_order.device)
        choices = plan.dispatch_order.index_select(0, send)
        receive_order, restore_order = _receive_maps(spec, send.device)
        source_end = source_begin + send.numel()
        chunks.append(DeviceChunk(spec, send, choices, choices // top_k,
                                  receive_order, restore_order,
                                  _upload(torch.tensor(spec.boundaries, dtype=torch.int64), send.device),
                                  source_begin, source_end))
        source_begin = source_end
    choices = torch.cat([chunk.source_choices for chunk in chunks]) if chunks else plan.dispatch_order.new_empty(0)
    inverse = torch.empty(choices.numel(), dtype=torch.int32, device=choices.device)
    inverse.scatter_(0, choices, torch.arange(choices.numel(), dtype=torch.int32, device=choices.device))
    chunks = tuple(chunks)
    restore = _expert_restore(plan, chunks, rank)
    identity = np.array_equal(restore, np.arange(len(restore)))
    expert_restore = None if identity else _upload(torch.from_numpy(restore), choices.device)
    positions = plan.unpermute_indices.reshape(-1, top_k)
    keys = positions.float() if positions.numel() < 2**24 else positions.long()
    choice_order = torch.argsort(keys, dim=-1)
    gradient_order = inverse.reshape_as(positions).gather(1, choice_order).reshape(-1)
    return ChunkLayout(chunks, inverse, expert_restore, gradient_order)
