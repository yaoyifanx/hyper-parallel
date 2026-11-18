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
    TokenDedup, _table, failed_local_dedup, finish_dedup, prepare_local_dedup,
)


@dataclass(frozen=True)
class GroupPlan:
    """Original choice identities and expert row maps shared by all packet phases."""

    dedup: TokenDedup
    source_forward: torch.Tensor
    source_backward: torch.Tensor
    target_forward: torch.Tensor
    target_backward: torch.Tensor
    row_slots: torch.Tensor
    score_bf16: bool
    receive_geometry: TileReceiveGeometry


def _source_members(local: Any, top_k: int, graphs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Build both phase member tables before the common metadata admission waits."""
    rows, ranks = local.source_choices.numel(), local.counts.shape[1]
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
    return _table(graphs, ('native_group_source', str(local.packet.device), rows, ranks, top_k), program,
                  local.source_choices, local.counts, source.choice_group, source.member_choice,
                  source.unique_counts, source.modes, local.forward_send, local.backward_send)


def compile_group_plan(plan: Any, executor: Any, hidden: torch.Tensor, scores: torch.Tensor,
                       key: Any, runtime: Any) -> GroupPlan:
    """Compose the existing expert wire geometry once, with no O(M) host downloads."""
    route, device, graphs = plan.route, hidden.device, runtime.planning_graphs
    source_rows, receive_rows, top_k = route.source_rows, route.received_rows, executor.top_k
    outgoing = sorted(route.outgoing, key=lambda edge: (edge.target, edge.source_row))
    incoming = sorted(route.incoming, key=lambda edge: (edge.source, edge.target_row))
    send, receive = [0] * route.ranks, [0] * route.ranks
    for edge in outgoing:
        send[edge.target] += edge.rows
    for edge in incoming:
        receive[edge.source] += edge.rows
    error = result = None
    try:
        source_members = None
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

            cache_key = ('native_group_wire', str(device), route.source_rows, route.received_rows,
                         len(outgoing), len(incoming), len(route.boundaries))
            if source_rows or receive_rows:
                source_wire, target_wire, row_slots = _table(graphs, cache_key, wire_maps,
                                                            source_spans, target_spans, lengths)
            else:
                source_wire = target_wire = torch.empty(0, dtype=torch.int32, device=device)
                row_slots = torch.empty((0, 2), dtype=torch.int32, device=device)
            order = plan.order.index_select(0, source_wire.long()) if source_rows else plan.order
            local = prepare_local_dedup(order, send, receive, executor.top_k,
                                        forward_ratio=executor.forward_ratio, backward_ratio=executor.backward_ratio,
                                        graphs=graphs, capabilities=32 + int(scores.dtype == torch.bfloat16),
                                        keep_groups=True)
            source_members = _source_members(local, executor.top_k, graphs)
        except Exception as exc:
            local_error = exc
            local = failed_local_dedup(send, receive, device=device, top_k=executor.top_k)
        dedup = finish_dedup(local, executor.group, device=device, width=hidden.shape[1], graphs=graphs,
                             execution=key, score_lanes=max(1, scores.element_size() // hidden.element_size()))
        if local_error is not None:
            raise local_error
        forward_rows, backward_rows = sum(dedup.forward_recv), sum(dedup.backward_recv)

        def expert_members(mapping: torch.Tensor, forward: torch.Tensor,
                           backward: torch.Tensor) -> tuple[torch.Tensor, ...]:
            """Translate both receive member tables from wire rows to expert rows."""
            if receive_rows == 0:
                return forward.reshape(0, top_k), backward.reshape(0, top_k)
            padded = torch.cat((mapping, mapping.new_full((1,), -1)))

            def transform(members: torch.Tensor) -> torch.Tensor:
                """Preserve absent members while applying the expert-row permutation."""
                indices = torch.where(members >= 0, members, receive_rows)
                return padded.index_select(0, indices.long()).reshape(-1, top_k)

            return transform(forward), transform(backward)

        member_key = ('native_group_target', str(device), route.received_rows,
                      forward_rows, backward_rows, executor.top_k)
        if receive_rows:
            target_forward, target_backward = _table(
                graphs, member_key, expert_members, target_wire,
                dedup.forward_target_members, dedup.backward_target_members)
        else:
            target_forward = target_backward = torch.empty((0, top_k), dtype=torch.int32, device=device)
        result = GroupPlan(dedup, source_members[0][:sum(dedup.forward_send)],
                           source_members[1][:sum(dedup.backward_send)], target_forward, target_backward,
                           row_slots, scores.dtype == torch.bfloat16,
                           TileReceiveGeometry(route.boundaries, device))
    except Exception as exc:
        error = exc
    runtime.workspace.check_preflight(error)
    return result
