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
"""Bind final expert placement to shared peer/token maps; activations stay out of plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hyper_parallel.core.multicore.modules.routed_experts.grouped import TileReceiveGeometry

from hyper_parallel.distributed.expert_parallel.moonep.peer_token_plan import (
    _member_table, _peer_rows, _prefix, _scatter_rows,
)
from hyper_parallel.distributed.expert_parallel.moonep.token_dedup import (
    TokenDedup, materialize_owned_tables, failed_local_dedup, finish_dedup, prepare_local_dedup,
)


@dataclass(frozen=True)
class GroupPlan:
    """Original choice identities and expert row maps shared by all packet phases."""

    token_transport: TokenDedup
    source_forward: torch.Tensor
    source_backward: torch.Tensor
    target_forward: torch.Tensor
    target_backward: torch.Tensor
    row_slots: torch.Tensor
    score_bf16: bool
    receive_geometry: TileReceiveGeometry


def _source_members(local: Any, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build both phase member tables before the common metadata admission waits."""
    rows = local.source_choices.numel()
    if rows == 0:
        empty = torch.empty((0, top_k), dtype=torch.int32, device=local.packet.device)
        return empty, empty

    def program(order: torch.Tensor, controls: torch.Tensor, groups: torch.Tensor,
                raw_members: torch.Tensor, unique: torch.Tensor, modes: torch.Tensor,
                forward: torch.Tensor, backward: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Map canonical members into both native source-phase row layouts."""
        positions = torch.arange(rows, dtype=torch.int32, device=order.device)
        peers = _peer_rows(controls[0], rows)
        raw = raw_members.reshape(-1)
        columns = torch.arange(top_k, dtype=torch.int32, device=order.device).expand(rows, top_k).reshape(-1)
        discarded = rows + torch.arange(raw.numel(), dtype=torch.int32, device=order.device)
        ordinal = _scatter_rows(positions.new_full((rows + raw.numel(),), -1),
                                 torch.where(raw >= 0, raw, discarded),
                                 torch.where(raw >= 0, columns, -1))[:rows]
        raw_slot = groups.index_select(0, order) - _prefix(unique).index_select(0, peers.long())
        dense = positions - _prefix(controls[0]).index_select(0, peers.long())

        def phase(counts: torch.Tensor, bit: int) -> torch.Tensor:
            """Compose one direction using the encoded peer modes and active group counts."""
            compact = (modes // bit % 2).index_select(0, peers.long()) != 0
            slots = _prefix(counts).index_select(0, peers.long()) + torch.where(compact, raw_slot, dense)
            member = torch.where(compact, ordinal.index_select(0, order), 0)
            return _member_table(slots, member, order.int(), rows, top_k)

        return phase(forward, 1), phase(backward, 4)

    source = local.source_groups
    return materialize_owned_tables(program,
                  local.source_choices, local.counts, source.choice_group, source.member_choice,
                  source.unique_counts, source.modes, local.forward_send, local.backward_send)


def compile_group_plan(plan: Any, executor: Any, hidden: torch.Tensor, scores: torch.Tensor,
                       key: Any, runtime: Any) -> GroupPlan:
    """Build expert wire geometry eagerly once, with no O(M) host downloads."""
    route, device = plan.route, hidden.device
    source_rows, receive_rows = route.source_rows, route.received_rows
    outgoing = sorted(route.outgoing, key=lambda edge: (edge.target, edge.source_row))
    incoming = sorted(route.incoming, key=lambda edge: (edge.source, edge.target_row))
    send, receive = [0] * route.ranks, [0] * route.ranks
    for edge in outgoing:
        send[edge.target] += edge.rows
    for edge in incoming:
        receive[edge.source] += edge.rows
    try:
        local_error = None
        try:
            source_spans = torch.tensor([[edge.source_row, edge.rows] for edge in outgoing],
                                        dtype=torch.int32, device='cpu').reshape(-1, 2).t().contiguous()
            target_spans = torch.tensor([[edge.target_row, edge.rows] for edge in incoming],
                                        dtype=torch.int32, device='cpu').reshape(-1, 2).t().contiguous()
            lengths = torch.tensor(
                [end - begin for begin, end in zip((0, *route.boundaries[:-1]), route.boundaries)],
                dtype=torch.int32, device='cpu')
            source_spans, target_spans, lengths = (
                tensor.pin_memory(device=device).to(device, non_blocking=True)
                for tensor in (source_spans, target_spans, lengths))

            def wire_maps(source: torch.Tensor, target: torch.Tensor,
                          expert_lengths: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Compose peer spans into source/expert rows and expert tile coordinates."""
                def expand(spans: torch.Tensor, total: int) -> torch.Tensor:
                    """Expand fixed-capacity peer intervals without downloading row coordinates."""
                    if total == 0:
                        return torch.empty(0, device=device, dtype=torch.int32)
                    owner = _peer_rows(spans[1], total).long()
                    return (spans[0] - _prefix(spans[1])).index_select(0, owner) + torch.arange(
                        total, dtype=torch.int32, device=device)

                source_order = expand(source, source_rows)
                expert_rows = expand(target, receive_rows)
                if receive_rows:
                    expert = _peer_rows(expert_lengths, receive_rows)
                    row = torch.arange(receive_rows, dtype=torch.int32, device=device)
                    tile = (row - _prefix(expert_lengths).index_select(0, expert.long())) // 128
                    slots = torch.stack((expert, tile), 1)
                else:
                    slots = torch.empty((0, 2), dtype=torch.int32, device=device)
                return source_order, expert_rows, slots

            if source_rows or receive_rows:
                source_wire, target_wire, row_slots = materialize_owned_tables(
                    wire_maps, source_spans, target_spans, lengths)
            else:
                source_wire = target_wire = torch.empty(0, dtype=torch.int32, device=device)
                row_slots = torch.empty((0, 2), dtype=torch.int32, device=device)
            order = plan.order.index_select(0, source_wire.long()) if source_rows else plan.order
            local = prepare_local_dedup(order, send, receive, executor.top_k,
                                        forward_ratio=executor.forward_ratio, backward_ratio=executor.backward_ratio,
                                        capabilities=32 + int(scores.dtype == torch.bfloat16),
                                        keep_groups=True)
        except Exception as exc:
            local_error = exc
            local = failed_local_dedup(send, receive, device=device, top_k=executor.top_k)
        prepared = {}

        def prepare_members() -> None:
            """Overlap local-only preparation with metadata, before common admission."""
            if local_error is not None:
                raise local_error
            prepared['members'] = _source_members(local, executor.top_k)
            prepared['geometry'] = TileReceiveGeometry(route.boundaries, device)

        dedup = finish_dedup(
            local, executor.group, device=device, width=hidden.shape[1],
            score_lanes=max(1, scores.element_size() // hidden.element_size()),
            target_row_values=target_wire if local_error is None else None, prepare=prepare_members)
        source_members = prepared['members']
        return GroupPlan(dedup, source_members[0][:sum(dedup.forward_send)],
                         source_members[1][:sum(dedup.backward_send)],
                         dedup.forward_target_members.reshape(-1, executor.top_k),
                         dedup.backward_target_members.reshape(-1, executor.top_k),
                         row_slots, scores.dtype == torch.bfloat16, prepared['geometry'])
    except Exception:
        # Every fallible plan preparation is covered by finish_dedup's common
        # admission. A second scalar all-reduce/host read adds no safety here.
        runtime.workspace.poisoned = True
        raise
