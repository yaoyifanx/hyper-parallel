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
"""Fixed-capacity peer/token grouping programs; payload vectors never enter them."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SourceGroupMap:
    """Original Top-K choices grouped by their final allocation target.

    Members contain original choice IDs, not dispatch row IDs. Active groups
    occupy a peer-major prefix; inactive capacity has -1 coordinates.
    """

    choice_group: torch.Tensor
    member_choice: torch.Tensor
    unique_counts: torch.Tensor
    modes: torch.Tensor


def _scatter_rows(output: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Write admitted narrow metadata rows using the CANN axis-scatter primitive."""
    if indices.numel() == 0:
        return output
    if output.device.type == "npu" and output.numel() < 4 * indices.numel():
        # Dense metadata favors narrow NdUpdate; sparse member capacity favors
        # axis scatter. This static geometry choice adds no routing decision.
        import torch_npu  # pylint: disable=import-outside-toplevel

        return torch_npu.npu_scatter_nd_update_(output, indices.int().reshape(-1, 1), values)
    return output.scatter_(0, indices.long(), values)


def _prefix(values: torch.Tensor) -> torch.Tensor:
    """Return exclusive INT32 offsets within the admitted selection capacity."""
    return values.cumsum(0, dtype=torch.int32) - values


def _peer_rows(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    """Expand bounded peer intervals with vector comparisons, without shape inference."""
    positions = torch.arange(capacity, device=counts.device, dtype=torch.int32)
    ends = counts.cumsum(0, dtype=torch.int32)
    return (positions[:, None] >= ends[None]).sum(-1, dtype=torch.int32)


def _member_table(groups: torch.Tensor, columns: torch.Tensor, values: torch.Tensor,
                  capacity: int, top_k: int) -> torch.Tensor:
    """Place unique group/member coordinates; invalid entries share a -1 dummy."""
    sentinel = capacity * top_k
    valid = (groups >= 0) & (groups < capacity) & (columns >= 0) & (columns < top_k)
    indices = torch.where(valid, groups * top_k + columns, sentinel)
    payload = torch.where(valid, values, -1)
    output = values.new_full((sentinel + 1,), -1)
    return _scatter_rows(output, indices, payload)[:-1].reshape(capacity, top_k)


def _source_members(wire_slots: torch.Tensor, inverse_order: torch.Tensor, top_k: int) -> torch.Tensor:
    """Order a token's distinct partials by wire position using only its K axis."""
    rows = wire_slots.numel()
    if rows == 0:
        return wire_slots.new_empty((0, top_k))
    ids = wire_slots.index_select(0, inverse_order).reshape(-1, top_k)
    slots = torch.arange(top_k, dtype=torch.int32, device=ids.device)
    same = ids[:, :, None] == ids[:, None, :]
    earlier = slots[None] < slots[:, None]
    first = ~(same & earlier).any(-1)
    keys = torch.where(first, ids, rows)
    lower = keys[:, None, :] < keys[:, :, None]
    tied = (keys[:, None, :] == keys[:, :, None]) & earlier
    rank = (lower | tied).sum(-1, dtype=torch.int32)
    positions = torch.arange(rows // top_k, device=ids.device, dtype=torch.int32)[:, None] * top_k + rank
    output = torch.empty_like(wire_slots)
    payload = torch.where(first, ids, -1).reshape(-1)
    return _scatter_rows(output, positions.reshape(-1), payload).reshape(-1, top_k)


def source_tables(order: torch.Tensor, controls: torch.Tensor, top_k: int,
                  capabilities: int = 0, *, keep_groups: bool = True) -> tuple[torch.Tensor, ...]:
    """Build a fixed-size metadata packet and both vector layouts on device.

    Args:
        order: Final target-major original choice IDs, a permutation of [0, T*K).
        controls: INT32 [3, R]: base peer counts, inclusive forward and backward
            compact limits. The host derives only these O(R) integer limits.
        top_k: Router fanout; controls the bounded comparison/member axis.
        capabilities: Agreed protocol and numeric policy tag carried with modes.
        keep_groups: Static consumer requirement for native group-member composition.

    Returns:
        Packet; fixed-capacity forward/backward token maps and counts; source
        finalize members and validity. Native consumers can also request group
        coordinates/members and modes. Dynamic counts stay on device until export.
    """
    rows, ranks = order.numel(), controls.shape[1]
    counts, forward_limit, backward_limit = controls.unbind(0)
    device = order.device
    positions = torch.arange(rows, device=device, dtype=torch.int32)
    rank_ids = torch.arange(ranks, device=device, dtype=torch.int32)
    triplet = torch.arange(3, device=device, dtype=torch.int32)
    peers = _peer_rows(counts, rows)
    choices = order.int()
    if rows:
        safe_order = choices.clamp(0, rows - 1)
        inverse = _scatter_rows(positions.new_full((rows,), -1), safe_order, positions)
        valid = ((choices >= 0) & (choices < rows)).all() & (inverse >= 0).all()
        inverse = inverse.clamp_min(0)
        target = peers.index_select(0, inverse).reshape(-1, top_k)
        wire_position = inverse.reshape(-1, top_k)
        slots = torch.arange(top_k, device=device, dtype=torch.int32)
        same = target[:, :, None] == target[:, None, :]
        first = ~(same & (slots[None] < slots[:, None])).any(-1)
        token_leaders = (first[:, :, None] & (target[:, :, None] == rank_ids)).sum(1, dtype=torch.int32)
        cumulative = token_leaders.cumsum(0, dtype=torch.int32)
        unique = token_leaders.sum(0, dtype=torch.int32)
        choice_group = (_prefix(unique).index_select(0, target.reshape(-1)).reshape_as(target)
                        + cumulative.gather(1, target.long()) - 1).reshape(-1)
        ordinal = (same & (wire_position[:, None, :] < wire_position[:, :, None])).sum(-1, dtype=torch.int32)
        raw_slots = choice_group.index_select(0, safe_order) - _prefix(unique).index_select(0, peers)
        raw_members = (_member_table(choice_group, ordinal.reshape(-1), positions, rows, top_k)
                       if keep_groups else None)
        wire_ordinal = ordinal.reshape(-1).index_select(0, safe_order)
    else:
        valid = torch.ones((), dtype=torch.bool, device=device)
        unique, raw_slots, inverse = counts.clone(), choices, choices
        choice_group, wire_ordinal = choices, choices
        raw_members = choices.new_empty((0, top_k)) if keep_groups else None
    forward = unique <= forward_limit
    backward = unique <= backward_limit
    forward_counts = torch.where(forward, unique, counts)
    backward_counts = torch.where(backward, unique, counts)
    dense_slots = positions - _prefix(counts).index_select(0, peers)

    def phase(compact: torch.Tensor, phase_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build this phase token map and the source-side partial reduction members."""
        use_compact = compact.index_select(0, peers)
        output_slot = _prefix(phase_counts).index_select(0, peers) + torch.where(
            use_compact, raw_slots, dense_slots)
        writer = ~use_compact | (wire_ordinal == 0)
        indices = torch.where(writer, output_slot, rows + positions)
        tokens = _scatter_rows(positions.new_zeros((2 * rows,)), indices, choices // top_k)[:rows]
        return tokens, _source_members(output_slot, inverse, top_k)

    forward_tokens, forward_members = phase(forward, forward_counts)
    backward_tokens, backward_members = phase(backward, backward_counts)
    modes = forward.int() * 3 + backward.int() * 12
    packet = positions.new_empty(3 * (rows + ranks))
    body_positions = (positions * 3 + peers * 3 + 3)[:, None] + triplet
    body = torch.stack((raw_slots, choices, wire_ordinal), 1)
    packet = _scatter_rows(packet, body_positions.reshape(-1), body.reshape(-1))
    headers = _prefix(counts + 1)[:, None] * 3 + triplet
    header_values = torch.stack((torch.where(valid, unique, -1),
                                 counts.new_full((ranks,), rows // top_k), modes + capabilities * 16), 1)
    packet = _scatter_rows(packet, headers.reshape(-1), header_values.reshape(-1))
    outputs = (packet, forward_tokens, backward_tokens, forward_counts, backward_counts,
               forward_members, backward_members)
    if keep_groups:
        outputs += (choice_group, raw_members, unique, modes)
    return (*outputs, valid)


def target_tables(packet: torch.Tensor, counts: torch.Tensor, rows: int,
                  top_k: int, capabilities: int = 0,
                  row_values: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """Decode peer headers and compose receive maps without any O(M) host data.

    Args:
        packet: Received INT32 triplets: per-peer (unique, source T, modes),
            followed by (raw group, original choice, member ordinal) per entry.
        counts: Base selection counts received from each source, already known.
        rows: Fixed receive-selection extent, equal to sum(counts).
        top_k: Fanout agreed at construction, also the member-table width.
        capabilities: Expected protocol and numeric policy tag from every peer.
        row_values: Optional validated wire-to-expert row permutation, avoiding a later member-table gather.

    Returns:
        Forward/backward counts, lookups/members and a device validity bit.
        Canonical coordinates are validated internally without retaining them.
    """
    ranks, device = counts.numel(), packet.device
    rank_ids = torch.arange(ranks, device=device, dtype=torch.int32)
    triplet = torch.arange(3, device=device, dtype=torch.int32)
    positions = torch.arange(rows, device=device, dtype=torch.int32)
    headers = _prefix(counts + 1)[:, None] * 3 + triplet
    header = packet.index_select(0, headers.reshape(-1)).reshape(ranks, 3)
    unique, source_tokens, encoded_modes = header.unbind(1)
    modes = encoded_modes % 16
    peers = _peer_rows(counts, rows)
    body_positions = (positions * 3 + peers * 3 + 3)[:, None] + triplet
    body = packet.index_select(0, body_positions.reshape(-1)).reshape(rows, 3)
    raw_slot, choices, ordinal = body.unbind(1)
    valid = ((unique >= 0) & (unique <= counts) & (source_tokens >= 0)
             & (encoded_modes // 16 == capabilities)).all()
    valid = valid & (((modes % 2) == (modes // 2 % 2))
                     & ((modes // 4 % 2) == (modes // 8 % 2))).all()
    valid = valid & ((raw_slot >= 0) & (raw_slot < unique.index_select(0, peers))
                     & (choices >= 0) & (choices < source_tokens.index_select(0, peers) * top_k)
                     & (ordinal >= 0) & (ordinal < top_k)).all()
    safe_unique = unique.clamp_min(0).minimum(counts)
    safe_slot = raw_slot.clamp_min(0).minimum((safe_unique.index_select(0, peers) - 1).clamp_min(0))
    raw_group = _prefix(safe_unique).index_select(0, peers) + safe_slot
    raw_members = _member_table(raw_group, ordinal, positions, rows, top_k)
    active = positions < safe_unique.sum(dtype=torch.int32)
    valid = valid & ((raw_members >= 0).sum(dtype=torch.int32) == rows)
    valid = valid & ((raw_members[:, 1:] < 0)
                     | ((raw_members[:, :-1] >= 0) & (raw_members[:, 1:] > raw_members[:, :-1]))).all()
    if rows:
        first_row = raw_members[:, 0].clamp(0, rows - 1)
        group_token = torch.where(active, choices.index_select(0, first_row) // top_k, -1)
        group_source = torch.where(active, peers.index_select(0, first_row), -1)
        safe_group = raw_group.clamp(0, rows - 1)
        valid = valid & ((~active) | (raw_members[:, 0] >= 0)).all()
        valid = valid & ((choices // top_k == group_token.index_select(0, safe_group))
                         & (peers == group_source.index_select(0, safe_group))).all()
    if row_values is not None:
        if (row_values.shape != (rows,) or row_values.device != device
                or row_values.dtype not in (torch.int32, torch.int64)):
            raise ValueError("row values must match the receive row permutation")
        valid = valid & ((row_values >= 0) & (row_values < rows)).all()
    dense = positions - _prefix(counts).index_select(0, peers)

    def phase(bit: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one phase into peer counts, selection lookups and reduction members."""
        compact = (modes // bit % 2) != 0
        receive = torch.where(compact, safe_unique, counts)
        use_compact = compact.index_select(0, peers)
        lookup = _prefix(receive).index_select(0, peers) + torch.where(use_compact, safe_slot, dense)
        values = positions if row_values is None else row_values.int()
        members = _member_table(lookup, torch.where(use_compact, ordinal, 0), values, rows, top_k)
        return receive, lookup, members

    forward_counts, forward_lookup, forward_members = phase(1)
    backward_counts, backward_lookup, backward_members = phase(4)
    return (forward_counts, backward_counts, forward_lookup, backward_lookup,
            forward_members, backward_members, valid)


def layout_indices(vectors: torch.Tensor, scores: torch.Tensor, width: int,
                   vector_capacity: int, score_total: int) -> tuple[torch.Tensor, ...]:
    """Compile fixed-capacity packet maps; the host later slices active prefixes.

    A dummy peer absorbs unused capacity, so repeat_interleave has an explicit
    output_size even though compact counts remain device values. No padded
    hidden-vector payload is allocated by this metadata program.
    """
    ranks, device = vectors.numel(), vectors.device
    tails = (scores + width - 1) // width
    splits = vectors + tails
    total_vectors = vectors.sum(dtype=torch.int32)
    tail_capacity = (score_total + width - 1) // width + ranks
    row_capacity = vector_capacity + tail_capacity
    vector_offsets, score_offsets = _prefix(vectors), _prefix(scores)
    row_offsets, tail_offsets = _prefix(splits), _prefix(tails)

    def repeat_peer(lengths: torch.Tensor, capacity: int) -> torch.Tensor:
        """Expand bounded peer intervals for a fixed-capacity packet axis."""
        return _peer_rows(lengths, capacity).long()

    def append(values: torch.Tensor) -> torch.Tensor:
        """Append the zero-valued dummy peer used by inactive capacity."""
        return torch.cat((values, values.new_zeros(1)))

    vector_peer = repeat_peer(vectors, vector_capacity)
    vector_local = torch.arange(vector_capacity, dtype=torch.int32, device=device) - append(
        vector_offsets).index_select(0, vector_peer)
    vector_rows = append(row_offsets).index_select(0, vector_peer) + vector_local
    score_peer = _peer_rows(scores, score_total).long()
    offset_dtype = (torch.int32 if row_capacity * width < 2**31 else torch.int64)
    score_local = torch.arange(score_total, device=device, dtype=offset_dtype) - score_offsets.index_select(
        0, score_peer).to(offset_dtype)
    score_elements = (row_offsets + vectors).index_select(0, score_peer).to(offset_dtype) * width + score_local
    packet_peer = repeat_peer(splits, row_capacity)
    local = torch.arange(row_capacity, device=device, dtype=torch.int32) - append(row_offsets).index_select(
        0, packet_peer)
    count = append(vectors).index_select(0, packet_peer)
    pack_order = torch.where(local < count, append(vector_offsets).index_select(0, packet_peer) + local,
                             total_vectors + append(tail_offsets).index_select(0, packet_peer) + local - count)
    score_positions = tail_offsets.index_select(0, score_peer).to(offset_dtype) * width + score_local
    return vector_rows, score_elements, pack_order, score_positions


def source_forward_tables(order: torch.Tensor, controls: torch.Tensor,
                          top_k: int) -> tuple[torch.Tensor, ...]:
    """Build a fixed-size X descriptor from producer-validated source choices.

    Args:
        order: Target-major permutation of original Top-K choice IDs.
        controls: INT32 [3,R] original send/receive counts and compact limits.
        top_k: Fixed fanout of the source model.

    Returns:
        Metadata packet, compact send counts, source slots and original token IDs.
        Token scattering may follow the independent packet submission.
    """
    rows, ranks = order.numel(), controls.shape[1]
    counts, limits = controls[0], controls[2]
    positions = torch.arange(rows, device=order.device, dtype=torch.int32)
    peers = _peer_rows(counts, rows)
    tokens = order.int() // top_k
    if rows:
        token_count = rows // top_k
        keys = ((peers * token_count + tokens).float() if ranks * token_count <= 2**24
                else peers.long() * token_count + tokens.long())
        indices = torch.argsort(keys, stable=True)
        sorted_keys = keys.index_select(0, indices)
        first = torch.cat((torch.ones(1, dtype=torch.bool, device=order.device),
                           sorted_keys[1:] != sorted_keys[:-1]))
        cumulative = first.cumsum(0, dtype=torch.int32)
        groups = _scatter_rows(positions.new_empty(rows), indices.int(), cumulative - 1)
        ends = counts.cumsum(0, dtype=torch.int32)
        peer_ends = cumulative.index_select(0, (ends - 1).clamp_min(0))
        peer_ends = torch.where(ends > 0, peer_ends, 0)
        unique = peer_ends - torch.cat((peer_ends.new_zeros(1), peer_ends[:-1]))
    else:
        groups, unique = positions, counts.clone()
    compact = unique <= limits
    forward = torch.where(compact, unique, counts)
    slots = _prefix(forward).index_select(0, peers) + torch.where(
        compact.index_select(0, peers), groups - _prefix(unique).index_select(0, peers),
        positions - _prefix(counts).index_select(0, peers))
    packet_indices = torch.cat((_prefix(counts + 1), positions + peers + 1))
    packet_values = torch.cat((forward, slots - _prefix(forward).index_select(0, peers)))
    packet = _scatter_rows(positions.new_empty(rows + ranks), packet_indices, packet_values)
    return packet, forward, slots, tokens


def target_forward_tables(packet: torch.Tensor, counts: torch.Tensor, rows: int,
                          order: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    """Decode bounded X slots into the consumer's expert-row coordinates.

    Args:
        packet: Source-peer headers followed by one local group slot per wire row.
        counts: Original receive counts on the same device.
        rows: Original receive extent admitted by the complete route.
        order: Optional producer-validated expert-to-wire permutation.

    Returns:
        Compact receive counts, expert input lookups and a device validity bit.
    """
    positions = torch.arange(rows, device=packet.device, dtype=torch.int32)
    groups = packet.index_select(0, _prefix(counts + 1))
    peers = _peer_rows(counts, rows)
    local = packet.index_select(0, positions + peers + 1)
    valid = ((groups >= 0) & (groups <= counts)).all()
    safe_groups = groups.clamp_min(0).minimum(counts)
    limits = safe_groups.index_select(0, peers)
    valid = valid & ((local >= 0) & (local < limits)).all()
    safe = local.clamp_min(0).minimum((limits - 1).clamp_min(0))
    lookup = _prefix(safe_groups).index_select(0, peers) + safe
    if order is not None:
        lookup = lookup.index_select(0, order)
    return groups, lookup, valid
