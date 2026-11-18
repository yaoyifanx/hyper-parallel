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
"""Device chunk maps; the host retains only exact HCCL splits and feature views."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterator

import torch

from hyper_parallel.distributed.expert_parallel.moonep.device_planning import expand_segments, inverse_permutation
from hyper_parallel.distributed.expert_parallel.moonep.planning import ExecutionPlan


@dataclass(frozen=True)
class ChunkSpec:
    """Host splits and physical expert-slot [begin, end) bounds for one round.

    arena_row_span is (begin, row_count) in canonical expert-token rows, not
    an exclusive end. Splits count token rows rather than tensor elements.
    """

    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    expert_slot_begin: int
    expert_slot_end: int
    arena_row_span: tuple[int, int]
    receive_identity: bool
    canonical_identity: bool


@dataclass(frozen=True)
class DeviceChunk:
    """Views of once-built device maps, reused by forward, replay and backward."""

    spec: ChunkSpec
    source_choices: torch.Tensor
    source_tokens: torch.Tensor
    receive_order: torch.Tensor | None
    restore_order: torch.Tensor | None
    boundaries: torch.Tensor
    source_begin: int
    source_end: int


@dataclass(frozen=True)
class ChunkLayout:
    """Shared transport maps and a canonical dW permutation with no tail upload."""

    chunks: tuple[DeviceChunk, ...]
    plan: ExecutionPlan
    top_k: int
    wire_inverse: torch.Tensor
    canonical_order: torch.Tensor | None
    source_choices: torch.Tensor
    source_tokens: torch.Tensor

    @cached_property
    def source_order(self) -> torch.Tensor:
        """Compose original choice positions with their packed source-cache rows."""
        return self.wire_inverse.index_select(0, self.plan.unpermute_indices).to(torch.int32)

    @cached_property
    def source_gradient_order(self) -> torch.Tensor:
        """Preserve the nonchunk TopK accumulation order without reading device values."""
        positions = self.plan.unpermute_indices.reshape(-1, self.top_k)
        # The original planner admits at most 2**20 choices, exactly representable in FP32.
        choice_order = torch.argsort(positions.float(), dim=-1)
        return self.source_order.reshape_as(positions).gather(1, choice_order).reshape(-1)

    def __iter__(self) -> Iterator[DeviceChunk]:
        """Visit chunks in their shared collective order."""
        return iter(self.chunks)

    def __len__(self) -> int:
        """Return the globally shared round count."""
        return len(self.chunks)

    def __getitem__(self, index: int) -> DeviceChunk:
        """Return one round's views."""
        return self.chunks[index]


def _specs(ranks: int, control: tuple) -> tuple[ChunkSpec, ...]:
    """Decode final small controls; perform no token-length indexing or planning on host."""
    result = []
    for row in control:
        send, receive = row[:ranks], row[ranks:2 * ranks]
        # Tail fields match compile_round_geometry: physical expert-slot range,
        # receive/canonical identities, then arena begin in expert-token rows.
        expert_slot_begin, expert_slot_end, receive_identity, canonical_identity, arena_row_begin = row[2 * ranks:]
        arena_row_span = arena_row_begin, sum(receive)
        result.append(ChunkSpec(tuple(send), tuple(receive), expert_slot_begin, expert_slot_end,
                                arena_row_span, bool(receive_identity), bool(canonical_identity)))
    return tuple(result)


def compile_chunks(plan: ExecutionPlan, chunk_rows: int, top_k: int, *,
                   backward_maps: bool = True) -> ChunkLayout:
    """Build all O(N) permutations on device; only retained small controls are decoded.

    Canonical expert rows use an inverse scatter, avoiding both the CPU tail
    reconstruction and the sort/argsort pair formerly used for dW restoration.
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    metadata = plan.chunk_metadata
    if metadata is None or metadata.chunk_rows != chunk_rows:
        raise ValueError('chunk_rows must be included in the original planning call')
    tables = metadata.tables
    specs = _specs(len(plan.token_send), metadata.control)
    sent, received = plan.dispatch_order.numel(), sum(plan.token_recv)
    all_send = expand_segments(tables.send_starts.transpose(1, 2), tables.send_rows.transpose(1, 2), sent)
    inverse_send = inverse_permutation(all_send)
    all_choices = plan.dispatch_order.index_select(0, all_send)
    all_tokens = all_choices // top_k
    receive_rows = tables.receive_rows.transpose(1, 2)
    sizes = tables.groups.sum(1, dtype=tables.groups.dtype)
    wire_bases = sizes.cumsum(0, dtype=sizes.dtype) - sizes
    global_receive = expand_segments(tables.receive_starts.transpose(1, 2) + wire_bases[:, None, None],
                                     receive_rows, received)
    row_bases = torch.repeat_interleave(wire_bases, sizes, output_size=received)
    receive_order = global_receive - row_bases
    restore_order = inverse_permutation(global_receive) - row_bases
    group_boundaries = tables.groups.cumsum(1, dtype=tables.groups.dtype).long()
    canonical = None
    if backward_maps and not all(spec.canonical_identity for spec in specs):
        canonical_rows = expand_segments(tables.canonical_starts.transpose(1, 2), receive_rows, received)
        canonical = inverse_permutation(canonical_rows)
    chunks, source_begin, receive_begin = [], 0, 0
    for ordinal, spec in enumerate(specs):
        source_end = source_begin + sum(spec.send_counts)
        receive_end = receive_begin + sum(spec.receive_counts)
        incoming = None if spec.receive_identity else receive_order[receive_begin:receive_end]
        outgoing = None if spec.receive_identity else restore_order[receive_begin:receive_end]
        chunks.append(DeviceChunk(spec, all_choices[source_begin:source_end],
                                  all_tokens[source_begin:source_end],
                                  incoming, outgoing,
                                  group_boundaries[ordinal, spec.expert_slot_begin:spec.expert_slot_end],
                                  source_begin, source_end))
        source_begin, receive_begin = source_end, receive_end
    return ChunkLayout(tuple(chunks), plan, top_k, inverse_send, canonical, all_choices, all_tokens)
