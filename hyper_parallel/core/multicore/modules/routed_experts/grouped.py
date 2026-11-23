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
"""Bounded peer/group packets with coupled vector/scalar publication and expert readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from hyper_parallel.core.multicore.modules.routed_experts.readiness import ExpertReadiness, upload_metadata
from hyper_parallel.core.multicore.modules.routed_experts.transport import _load_native, vector_core_count
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingWorkspace


@dataclass
class GroupExchange:
    """Own one admitted packet phase, including outputs, producer fences and sequences."""

    workspace: RingWorkspace
    value: torch.Tensor
    scalars: torch.Tensor
    inputs: tuple[torch.Tensor, ...]
    tables: tuple[torch.Tensor, ...]
    config: tuple[int, ...]
    progress: torch.Tensor
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]
    initialized: Any
    source_admitted: Any
    complete: Any
    dependency: ExpertReadiness | None
    ready: ExpertReadiness | None
    generation: int
    submitted: bool = False

    def wait(self) -> torch.Tensor:
        """Join the completed vector and scalar message at the source consumer."""
        if not self.submitted:
            raise RuntimeError('group exchange must be submitted before consumption')
        current = torch.npu.current_stream(self.value.device)
        current.wait_event(self.complete)
        self.value.record_stream(current)
        self.scalars.record_stream(current)
        return self.value


class TileReceiveGeometry:
    """Cache immutable tile expectations; every invocation receives fresh counters."""

    def __init__(self, boundaries: Sequence[int], device: torch.device) -> None:
        """Bind expert boundaries and cache device-side per-tile expectations."""
        lengths = torch.tensor([end - begin for begin, end in zip((0, *boundaries[:-1]), boundaries)],
                               dtype=torch.int32, device='cpu')
        tiles = (lengths + 127) // 128
        producers = max(1, int(tiles.max()) if tiles.numel() else 1)
        expected = (lengths[None]
                    - torch.arange(producers, dtype=torch.int32, device='cpu')[:, None] * 128).clamp(0, 128)
        experts = torch.repeat_interleave(torch.arange(len(boundaries), device='cpu'), tiles.long())
        offsets = tiles.cumsum(0, dtype=torch.int32) - tiles
        local = torch.arange(experts.numel(), device='cpu') - offsets.index_select(0, experts)
        per_task = ((torch.arange(producers, device='cpu')[:, None] == local[None]).int()
                    * expected[local, experts][None])
        self.device, self.expected = device, expected
        self.target = upload_metadata(expected, device)
        self.tasks = upload_metadata(per_task.contiguous(), device)
        self.initialized = torch.npu.Event()
        self.initialized.record(torch.npu.current_stream(device))

    def new_fence(self) -> ExpertReadiness:
        """Require all rows of a tile, even when group expansion writes them out of order."""
        torch.npu.current_stream(self.device).wait_event(self.initialized)
        return ExpertReadiness.create(self.expected, self.device, deferred_poll=True,
                                       expected_device=self.target, task_expected=self.tasks)


def prepare_group_exchange(value: torch.Tensor, send_members: torch.Tensor, receive_members: torch.Tensor,
                           send_counts: Sequence[int], receive_counts: Sequence[int],
                           row_slots: torch.Tensor, workspace: RingWorkspace, *, phase: int,
                           source_tokens: torch.Tensor | None = None, scores: torch.Tensor | None = None,
                           target_scores: torch.Tensor | None = None, saved: torch.Tensor | None = None,
                           output_rows: int, scalar_rows: int = 0, score_bf16: bool = False,
                           dependency: ExpertReadiness | None = None, ready: ExpertReadiness | None = None,
                           output: torch.Tensor | None = None) -> GroupExchange:
    """Prepare all storage before phase admission; metadata and split counts are reused.

    Phase 1 sends X/prob and expands expert inputs; phase 2 returns FP32 weighted
    partials; phase 3 sends G and creates D/dProb; phase 4 returns FP32 dX partials
    with independent per-selection dProb. Replica payloads are outside this API.
    """
    if phase not in (1, 2, 3, 4) or workspace.closed or workspace.poisoned or workspace.closing:
        raise ValueError('group phase or workspace is unavailable')
    _load_native()
    device, width = value.device, value.shape[1]
    ranks, top_k = workspace.spec.ranks, send_members.shape[1]
    send_counts, receive_counts = tuple(send_counts), tuple(receive_counts)
    if (len(send_counts) != ranks or len(receive_counts) != ranks
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                   for count in (*send_counts, *receive_counts))
            or sum(send_counts) != send_members.shape[0] or sum(receive_counts) != receive_members.shape[0]
            or send_counts[workspace.rank] != receive_counts[workspace.rank]):
        raise ValueError('group counts must partition source/target members, including identical self traffic')
    direction = int(phase in (2, 4))
    scalar_stride = (top_k + 7) // 8 * 32
    vector_bytes = 4 if phase in (2, 4) or value.dtype == torch.float32 else 2
    scalar_bytes = scalar_stride if phase in (1, 4) else 0
    quantum = min(64, (workspace.spec.block_rows * width * 2 - 63) // (width * vector_bytes + scalar_bytes))
    if quantum < 1:
        raise ValueError('one vector/scalar group cannot fit the admitted symmetric window')
    generation = workspace.metadata.get(('transport_generation', direction), 0)
    source_base, target_base = 0, 0
    geometry = []
    for peer, (outgoing, incoming) in enumerate(zip(send_counts, receive_counts)):
        first_send = workspace.send_sequence[direction][peer]
        first_receive = workspace.receive_sequence[direction][peer]
        if (min(first_send, first_receive) < 0
                or max(first_send + (outgoing + quantum - 1) // quantum,
                       first_receive + (incoming + quantum - 1) // quantum) >= 2**30):
            raise RuntimeError('group sequences need a drained reset before reuse')
        geometry.append((source_base, outgoing, target_base, incoming, first_send, first_receive, 0, 0))
        source_base += outgoing
        target_base += incoming
    empty_score = torch.empty(0, dtype=torch.float32, device=device)
    inputs = (value, empty_score if scores is None else scores,
              empty_score if target_scores is None else target_scores,
              torch.empty((0, width), dtype=torch.bfloat16, device=device) if saved is None else saved)
    source_tokens = torch.empty(0, dtype=torch.int32, device=device) if source_tokens is None else source_tokens
    tables = (source_tokens, send_members, receive_members,
              upload_metadata(torch.tensor(geometry, dtype=torch.int64, device='cpu'), device), row_slots)
    dtype = torch.bfloat16 if phase in (1, 3) else torch.float32
    output = torch.empty((output_rows, width), dtype=dtype, device=device) if output is None else output
    scalars = torch.empty(scalar_rows, dtype=torch.float32, device=device)
    progress = torch.zeros((ranks, workspace.spec.lanes, 16), dtype=torch.int64, device=device)
    experts = (ready.flags.shape[1] if ready is not None else
               dependency.flags.shape[1] if dependency is not None else 0)
    budget = vector_core_count() - 8
    workers = ranks * workspace.spec.lanes if ranks * workspace.spec.lanes <= budget else min(ranks, budget)
    config = (phase, workspace.rank, ranks, workspace.spec.lanes, workspace.spec.block_rows, quantum,
              generation + 1, int(score_bf16), workers, 60_000_000_000,
              experts, ready.flags.shape[0] if ready is not None else 0)
    torch.ops.hyper_parallel.routed_experts_group_exchange_validate(
        inputs, (output, scalars), tables, workspace.data[direction], workspace.signals[direction],
        workspace.headers[direction], progress, config,
        None if dependency is None else dependency.flags, None if dependency is None else dependency.expected,
        None if ready is None else ready.flags)
    initialized = torch.npu.Event()
    initialized.record(torch.npu.current_stream(device))
    return GroupExchange(workspace, output, scalars, inputs, tables, config, progress, send_counts, receive_counts,
                         initialized, torch.npu.Event(), torch.npu.Event(), dependency, ready, generation)


def submit_group_exchange(work: GroupExchange) -> GroupExchange:
    """Queue one phase on the existing EP stream; no allocation or collective on T."""
    workspace = work.workspace
    direction = int(work.config[0] in (2, 4))
    generation_key = ('transport_generation', direction)
    if work.submitted or workspace.metadata.get(generation_key, 0) != work.generation:
        raise RuntimeError('group exchange was already submitted or its captured sequences are stale')
    current = torch.npu.current_stream(work.value.device)
    if work.dependency is None:
        work.source_admitted.record(current)
    else:
        work.dependency.validate_polling()
    with torch.npu.stream(workspace.transport_stream):
        with workspace.submission(work.value.device):
            stream = workspace.transport_stream
            stream.wait_event(work.initialized)
            if work.dependency is None:
                stream.wait_event(work.source_admitted)
            else:
                stream.wait_event(work.dependency.admitted)
                stream.wait_event(work.dependency.poll_after.event)
            if work.ready is not None:
                work.ready.poll_after.event.record(stream)
            work.submitted = True
            torch.ops.hyper_parallel.routed_experts_group_exchange(
                work.inputs, (work.value, work.scalars), work.tables, workspace.data[direction],
                workspace.signals[direction], workspace.headers[direction], work.progress, work.config,
                None if work.dependency is None else work.dependency.flags,
                None if work.dependency is None else work.dependency.expected,
                None if work.ready is None else work.ready.flags)
            if work.ready is not None:
                work.ready.poll_after.queued = True
            quantum = work.config[5]
            for peer, (send, receive) in enumerate(zip(work.send_counts, work.receive_counts)):
                if peer != workspace.rank:
                    workspace.send_sequence[direction][peer] += (send + quantum - 1) // quantum
                    workspace.receive_sequence[direction][peer] += (receive + quantum - 1) // quantum
            workspace.metadata[generation_key] = work.generation + 1
            workspace.last_progress = work.progress
            work.complete.record(stream)
    if work.ready is not None:
        work.value.record_stream(current)
        work.scalars.record_stream(current)
    return work
