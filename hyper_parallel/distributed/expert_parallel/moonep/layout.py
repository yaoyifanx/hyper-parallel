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
"""Device interval geometry shared by complete and chunked MoonEP transport."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from hyper_parallel.distributed.expert_parallel.moonep.device_planning import TableGraph, peer_round_quotas


@dataclass(frozen=True)
class RouteGeometry:
    """Small device tables; active home slots precede a sorted replica suffix."""

    source_starts: torch.Tensor
    target_ends: torch.Tensor
    send_rows: torch.Tensor
    receive_rows: torch.Tensor
    peer_rows: torch.Tensor
    active: torch.Tensor
    groups: torch.Tensor
    weight_send: torch.Tensor
    weight_recv: torch.Tensor
    send_ids: torch.Tensor
    max_receive_rows: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Expose graph inputs without dataclasses.astuple deep-copying tensor storage."""
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class RoundGeometry:
    """Tensor intervals for all rounds, without any token-length CPU arrays."""

    send_starts: torch.Tensor
    send_rows: torch.Tensor
    receive_starts: torch.Tensor
    receive_rows: torch.Tensor
    canonical_starts: torch.Tensor
    groups: torch.Tensor
    control: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return graph outputs in the dataclass constructor's stable field order."""
        return tuple(getattr(self, field.name) for field in fields(self))


@dataclass(frozen=True)
class ChunkMetadata:
    """Device intervals and exact Host splits exported with the original plan."""

    chunk_rows: int
    tables: RoundGeometry
    control: tuple[tuple[int, ...], ...]


def compile_round_geometry(route: RouteGeometry, rank: int, chunk_rows: int,
                           local_rows: int, graphs: dict | None = None) -> RoundGeometry:
    """Compile round controls before the plan's one Host export.

    Args:
        route: Device placement intervals from the original counts collective.
        rank: EP-local rank.
        chunk_rows: Positive maximum received entries per round.
        local_rows: Host-known source extent used only to size a graph bucket.
        graphs: Shared topology-program cache, if the caller uses compiled NPU tables.

    Returns:
        Fixed-capacity device intervals. The exported plan must check that the
        actual round count fits before using any splits; unequal source sizes
        may require growing the bucket on that cold path.
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    extent = route.max_receive_rows
    if extent.device.type == 'cpu':
        rounds = (int(extent) + chunk_rows - 1) // chunk_rows
        return round_geometry(route, rank, rounds)
    if graphs is None:
        raise ValueError('device chunk planning requires a topology-program cache')
    ranks, experts = route.receive_rows.shape
    key = ('rounds', str(extent.device), ranks, experts, route.active.numel(), rank, chunk_rows)
    estimate = (local_rows + chunk_rows - 1) // chunk_rows
    required = max(1, 1 << (max(1, estimate) - 1).bit_length())
    previous = graphs.get(key)
    if previous is None or previous.capacity < required:
        def program(actual, *parts):
            return round_geometry(RouteGeometry(*parts), rank, required, valid_rounds=actual).tensors()

        graphs[key] = TableGraph(program, extent, capacity=required)
    if chunk_rows > torch.iinfo(extent.dtype).max - 2**20:
        extent = extent.long()
    actual = (extent + chunk_rows - 1) // chunk_rows
    return RoundGeometry(*graphs[key](actual, *route.tensors()))


def route_geometry(counts: torch.Tensor, allocation: torch.Tensor, rank: int, capacity: int) -> RouteGeometry:
    """Intersect source/expert and expert/destination intervals entirely on device."""
    ranks, experts = counts.shape
    home_count = experts // ranks
    # The executor admits at most 2**20 entries per source. Keep counter
    # prefixes on the native vector path; INT64 Cumsum otherwise uses AiCPU.
    if counts.device.type == 'npu' and ranks * 2**20 < 2**31:
        counts, allocation = counts.int(), allocation.int()
    source_ends = counts.cumsum(0, dtype=counts.dtype)
    target_ends = allocation.cumsum(1, dtype=allocation.dtype)
    source_starts, target_starts = source_ends - counts, target_ends - allocation
    overlap = (torch.minimum(source_ends[:, :, None], target_ends[None])
               - torch.maximum(source_starts[:, :, None], target_starts[None])).clamp_min(0)
    send_rows, receive_rows = overlap[rank].clone(), overlap[:, :, rank].clone()
    peer_rows = overlap.sum(1, dtype=overlap.dtype)
    expert_ids = torch.arange(experts, device=counts.device)
    rank_ids = torch.arange(ranks, device=counts.device)
    remote = (allocation > 0) & (expert_ids[:, None] // home_count != rank_ids[None])
    remote_here = remote[:, rank]
    order = torch.argsort((expert_ids + (~remote_here).long() * experts).float())[:capacity]
    active = torch.cat((expert_ids[rank * home_count:(rank + 1) * home_count], order))
    enabled = torch.cat((torch.ones(home_count, device=counts.device, dtype=torch.bool), remote_here[order]))
    groups = allocation[:, rank].index_select(0, active) * enabled
    weight_send = remote[rank * home_count:(rank + 1) * home_count].sum(0)
    weight_recv = remote[:, rank].reshape(ranks, home_count).sum(1)
    positions = torch.arange(experts, device=counts.device)
    send_enabled = remote[rank * home_count:(rank + 1) * home_count].T.reshape(-1)
    send_order = torch.argsort((positions + (~send_enabled).long() * experts).float())
    return RouteGeometry(source_starts, target_ends, send_rows, receive_rows, peer_rows,
                         active, groups, weight_send, weight_recv, send_order % home_count,
                         allocation.sum(0, dtype=allocation.dtype).max())


def round_geometry(route: RouteGeometry, rank: int, rounds: int, *,
                   valid_rounds: torch.Tensor | None = None) -> RoundGeometry:
    """Compile peer splits, GMM groups and canonical/arena intervals once per plan."""
    ranks, experts = route.receive_rows.shape
    active = route.active
    slots = active.numel()
    quotas = peer_round_quotas(route.peer_rows, rounds, valid_rounds)
    send = quotas[:, rank]
    receive = quotas[:, :, rank]
    sent_before = send.cumsum(0, dtype=send.dtype) - send
    received_before = receive.cumsum(0, dtype=receive.dtype) - receive
    send_positions = route.send_rows.cumsum(0, dtype=route.send_rows.dtype) - route.send_rows
    outgoing_first = torch.maximum(send_positions[None], sent_before[:, None])
    outgoing = (torch.minimum((send_positions + route.send_rows)[None], (sent_before + send)[:, None])
                - outgoing_first).clamp_min(0)
    peer_bases = route.peer_rows[rank].cumsum(0, dtype=route.peer_rows.dtype) - route.peer_rows[rank]
    send_starts = peer_bases[None, None] + outgoing_first
    receive_positions = route.receive_rows.cumsum(1, dtype=route.receive_rows.dtype) - route.receive_rows
    incoming_first = torch.maximum(receive_positions[None], received_before[:, :, None])
    raw_incoming = (torch.minimum((receive_positions + route.receive_rows)[None],
                                 (received_before + receive)[:, :, None]) - incoming_first).clamp_min(0)
    wire_starts = (raw_incoming.reshape(rounds, ranks * experts).cumsum(1, dtype=raw_incoming.dtype)
                   .reshape_as(raw_incoming) - raw_incoming)
    incoming = raw_incoming.index_select(2, active) * (route.groups > 0)[None, None]
    receive_starts = wire_starts.index_select(2, active)
    groups = incoming.sum(1, dtype=incoming.dtype)
    group_bases = route.groups.cumsum(0, dtype=route.groups.dtype) - route.groups
    canonical = (group_bases[None, None]
                 + (route.receive_rows.cumsum(0, dtype=route.receive_rows.dtype)
                    - route.receive_rows).index_select(1, active)[None]
                 + (incoming_first - receive_positions[None]).index_select(2, active))
    group_ends = groups.cumsum(1, dtype=groups.dtype)
    source_first = group_ends - groups
    round_sizes = groups.sum(1, dtype=groups.dtype)
    round_bases = round_sizes.cumsum(0, dtype=round_sizes.dtype) - round_sizes
    target_first = round_bases[:, None] + source_first
    arena = target_first[:, None] + incoming.cumsum(1, dtype=incoming.dtype) - incoming
    slot_ids = torch.arange(slots, device=groups.device)[None].expand(rounds, slots)
    # Slot labels are exact in FP32 at the admitted expert limit. Native
    # MaxDim avoids unavailable amin/all graph converters and INT64 reducers.
    low = torch.where(groups > 0, slot_ids, slots).float().neg().max(1).values.neg().long()
    high = torch.where(groups > 0, slot_ids + 1, 0).float().max(1).values.long()
    low = torch.where(high > 0, low, 0)
    lengths = incoming.transpose(1, 2).reshape(rounds, slots * ranks)
    desired_starts = lengths.cumsum(1, dtype=lengths.dtype) - lengths
    wire = receive_starts.transpose(1, 2).reshape_as(lengths)
    receive_identity = (((wire != desired_starts).int() * (lengths != 0).int()).sum(1) == 0)
    canonical_identity = (((canonical != arena).int() * (incoming != 0).int()).sum((1, 2)) == 0)
    # Host row: peer send/receive token counts, expert-slot begin/end,
    # receive/canonical identity flags, then the arena begin in token rows.
    control = torch.cat((send, receive,
                         torch.stack((low, high, receive_identity, canonical_identity, round_bases), 1)), 1)
    control = control.to(groups.dtype)
    return RoundGeometry(send_starts, outgoing, receive_starts, incoming, canonical, groups, control)
