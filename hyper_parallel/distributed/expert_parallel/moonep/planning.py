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
"""MoonEP 容量规划及一次性通信元数据编译；大索引始终留在设备。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch

from hyper_parallel.distributed.expert_parallel.moonep.device_planning import (
    HostControl, candidate_metrics, expand_segments,
)
from hyper_parallel.distributed.expert_parallel.moonep.layout import (
    ChunkMetadata, RouteGeometry, compile_round_geometry, route_geometry,
)
from hyper_parallel.distributed.expert_parallel.moonep.device_dispatch import NativeDispatchPlan
from hyper_parallel.distributed.expert_parallel.moonep.token_dedup import ForwardTokenDedup, TokenDedup


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable routes with fixed home slots followed by contiguous replica slots."""

    dispatch_order: torch.Tensor
    unpermute_indices: torch.Tensor
    sort_order: torch.Tensor | None
    boundaries: torch.Tensor
    group_counts: tuple[int, ...]
    token_send: list[int]
    token_recv: list[int]
    weight_send: list[int]
    weight_recv: list[int]
    send_ids: torch.Tensor
    home_count: int
    any_migration: bool
    metrics: tuple[int, ...]
    token_transport: TokenDedup | ForwardTokenDedup | NativeDispatchPlan | None = None
    send_slots: tuple[int, ...] = ()
    planner_fallback: bool = False
    chunk_metadata: ChunkMetadata | None = None


def _stable_order(ids: torch.Tensor, labels: int) -> torch.Tensor:
    """Sort exact labels stably without encoding entry positions into FP32 keys."""
    if ids.numel() == 0:
        return torch.empty(0, device=ids.device, dtype=torch.long)
    # Label range is host topology metadata. Entry count no longer restricts
    # the FP32 fast path; stable sorting preserves every equal-label row.
    keys = ids.float() if labels <= 2**24 else ids
    return torch.argsort(keys, stable=True)


def prepare_entry_order(ids: torch.Tensor, experts: int) -> torch.Tensor:
    """按专家和原选择位置排序，供后续目标卡排序复用。

    Args:
        ids: 本卡展开后的专家号，例如 [2,0,2]。
        experts: 全局专家数；该例至少为三。

    Returns:
        设备上的专家主序索引；该例为 [1,0,2]。

    Example:
        [2,0,2] 返回 [1,0,2]，目标卡排序再复用这个顺序。
    """
    return _stable_order(ids, experts)


def _entry_order(counts: torch.Tensor, allocation: torch.Tensor,
                 rank: int, ids: torch.Tensor, metadata: torch.Tensor | None = None,
                 entry_order: torch.Tensor | None = None) -> torch.Tensor:
    """按计算卡、专家、原选择编号排列发送项；例如四项得到[0,2,1,3]。

    Args:
        counts: Source/expert device counts used for routing prefixes.
        allocation: Expert/destination device allocation.
        rank: EP组内源编号。
        ids: 设备上的本地展开专家号，绝不搬到CPU排序。
        metadata: 可复用的源／目标区间前缀；未提供时在设备生成。
        entry_order: counts通信期间提前生成的稳定条目排列；未提供时在此生成。

    Returns:
        设备上的发送排列；每个索引指向一个token的一个top-k选择。
    """
    size = ids.numel()
    experts, ranks = allocation.shape
    positions = torch.arange(size, device=ids.device)
    if entry_order is None:
        entry_order = prepare_entry_order(ids, experts)
    # Validation is read back asynchronously. Keep its speculative lookups
    # in bounds so malformed routes still fail at the common control check.
    sorted_ids = ids.index_select(0, entry_order).clamp(0, experts - 1)
    if metadata is None:
        offset = counts[:rank].sum(0) - (counts[rank].cumsum(0) - counts[rank])
        metadata = torch.cat((offset, allocation.cumsum(1).reshape(-1))).to(ids.device)
    offset, ends = metadata[:experts], metadata[experts:].reshape(experts, ranks)
    ordinal = positions + offset.index_select(0, sorted_ids)
    destinations = (ordinal[:, None] >= ends.index_select(0, sorted_ids)).sum(1)
    return entry_order.index_select(0, _stable_order(destinations, ranks))


def _cpu_transfer_quotas(home_totals: np.ndarray, targets: np.ndarray, fill: bool) -> np.ndarray:
    """Preserve first-rank tie breaking without batch-index arrays inside the small greedy loop."""
    ranks = home_totals.shape[0]
    balance = (home_totals.sum(axis=1) - targets).tolist()
    quotas = np.zeros((ranks, ranks), dtype=np.int64)
    rank_ids = range(ranks)
    for _ in range(ranks - 1):
        source = max(rank_ids, key=balance.__getitem__)
        target = min(rank_ids, key=balance.__getitem__)
        amount = -balance[target] if fill else min(balance[source], -balance[target])
        quotas[source, target] += amount
        balance[source] -= amount
        balance[target] += amount
    return quotas


def _cpu_candidates(totals: np.ndarray, ranks: int, rules: tuple[bool, ...]) -> np.ndarray:
    """Generate deterministic candidates using bounded scalar rounds on the CPU small table."""
    home_count = totals.size // ranks
    rank_ids = np.arange(ranks, dtype=np.int64)
    total = int(totals.sum())
    targets = total // ranks + (rank_ids < total % ranks).astype(np.int64)
    home_totals = totals.reshape(ranks, home_count)
    expert_order = np.argsort(-home_totals.astype(np.float32), axis=1, kind='stable')
    ordered_experts = np.take_along_axis(home_totals, expert_order, axis=1)
    expert_ends = ordered_experts.cumsum(axis=1)[:, :, None]
    candidates = []
    for fill in rules:
        quotas = _cpu_transfer_quotas(home_totals, targets, fill)
        target_order = np.argsort(-quotas.astype(np.float32), axis=1, kind='stable')
        ordered_quotas = np.take_along_axis(quotas, target_order, axis=1)
        target_ends = ordered_quotas.cumsum(axis=1)[:, None, :]
        moved_ordered = np.maximum(np.minimum(expert_ends, target_ends)
                                  - np.maximum(expert_ends - ordered_experts[:, :, None],
                                               target_ends - ordered_quotas[:, None, :]), 0)
        by_expert = np.empty_like(moved_ordered)
        np.put_along_axis(by_expert, target_order[:, None, :], moved_ordered, axis=2)
        moved = np.empty_like(by_expert)
        np.put_along_axis(moved, expert_order[:, :, None], by_expert, axis=1)
        remaining = home_totals - moved.sum(axis=2)
        moved[rank_ids, :, rank_ids] += remaining
        candidates.append(moved.reshape(totals.size, ranks))
    return np.stack(candidates)


def _cpu_candidate_metrics(allocation: np.ndarray, totals: np.ndarray, capacity: int) -> np.ndarray:
    """批量验证 CPU 候选及最小迁移证书；例如两张表返回 [2,6] 指标。"""
    batches, experts, ranks = allocation.shape
    home_count = experts // ranks
    home_ids = np.arange(experts) // home_count
    remote = home_ids[:, None] != np.arange(ranks)[None, :]
    enabled = (allocation > 0) & remote
    incoming = enabled.sum(axis=1)
    outgoing = enabled.reshape(batches, ranks, home_count, ranks).sum(axis=(2, 3))
    total = int(totals.sum())
    targets = total // ranks + (np.arange(ranks) < total % ranks)
    valid = ((allocation >= 0).all(axis=(1, 2)) & (totals >= 0).all() & (capacity >= 0)
             & (allocation.sum(axis=2) == totals).all(axis=1)
             & (allocation.sum(axis=1) == targets).all(axis=1)
             & (incoming <= capacity).all(axis=1))
    moved = (allocation * remote).sum(axis=(1, 2))
    peak = np.maximum(incoming.max(axis=1), outgoing.max(axis=1))
    copies = enabled.sum(axis=(1, 2))
    lower_bound = np.maximum(totals.reshape(ranks, home_count).sum(axis=1) - targets, 0).sum()
    certificate = valid & (moved == lower_bound)
    return np.stack((valid, moved, peak, copies, np.broadcast_to(lower_bound, (batches,)),
                     certificate), axis=1).astype(np.int64)


def select_allocation(counts: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """在 CPU 小计数表上选可行规划，优先采用达到迁移下界的候选。

    Args:
        counts: int64[R,E] 全局源计数；每个 EP rank 从同一次 all-gather 获得。
        capacity: 每卡最多接收的不同远程专家数，默认由调用方设为 E/R。

    Returns:
        CPU 计数表、CPU 分配表、是否使用填满接收卡兜底；逐条路由仍留在设备。

    Raises:
        ValueError: 计数形状、数值或容量不合法。
        RuntimeError: 两个候选均不能同时满足均衡与副本容量。

    Example:
        R=4、E=8、B=2、专家总量为 [0,0,4,4,4,4,4,4] 时，
        部分填充超出副本容量，采用填满接收卡候选，迁移 12 条。
    """
    if counts.ndim != 2 or counts.dtype != torch.int64:
        raise ValueError("MoonEP计数必须是int64[R,E]")
    ranks, experts = counts.shape
    if ranks < 1 or experts < 1 or experts % ranks:
        raise ValueError("MoonEP计数的rank数和专家数必须为正，且专家数能被rank数整除")
    if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 0:
        raise ValueError("MoonEP副本容量必须是非负整数")

    # This synchronous interface serves CPU/native-descriptor consumers.
    # MoonEPExecutor keeps its histogram on device and uses the tensor compiler.
    cpu = counts.cpu()
    source_counts = cpu.numpy()
    if np.any(source_counts < 0):
        raise ValueError("MoonEP源计数必须非负")
    totals = source_counts.sum(axis=0)
    if ranks == 1:
        return cpu, torch.from_numpy(totals.reshape(experts, 1).copy()), False

    # 自然路由通常由第一条候选直接获得 M=M0 证书。只在未获证书时
    # 生成填满接收卡候选，避免每层无条件构造第二张 [E,R] 小表。
    first = _cpu_candidates(totals, ranks, (False,))
    first_score = _cpu_candidate_metrics(first, totals, capacity)
    if first_score[0, 5]:
        return cpu, torch.from_numpy(first[0].copy()), False
    if ranks == 2:
        fallback, fallback_score = first[0], first_score[0]
    else:
        second = _cpu_candidates(totals, ranks, (True,))
        fallback = second[0]
        fallback_score = _cpu_candidate_metrics(second, totals, capacity)[0]
    if not fallback_score[0]:
        raise RuntimeError(f"MoonEP快速规划无法满足均衡及副本容量 B={capacity}；"
                           f"候选指标={[first_score[0].tolist(), fallback_score.tolist()]}")
    return cpu, torch.from_numpy(fallback.copy()), True


def compile_plan(counts: torch.Tensor, allocation: torch.Tensor, ids: torch.Tensor,
                 rank: int, *, build_receive_order: bool = True,
                 replica_capacity: int | None = None,
                 metrics: torch.Tensor | None = None, valid_routes: torch.Tensor | None = None,
                 entry_order: torch.Tensor | None = None, geometry: RouteGeometry | None = None,
                 chunk_rows: int | None = None, graphs: dict | None = None) -> ExecutionPlan:
    """Compile device geometry, then export only host collective/slicing controls.

    Args:
        counts: Source-rank/expert histogram, retained on the routing device.
        allocation: Expert/destination placement, retained on that same device.
        ids: This rank's original flattened expert choices.
        rank: EP-local source and receiver rank.
        build_receive_order: Build the complete receive permutation for nonchunk consumers.
        replica_capacity: Maximum remote slots; None permits every remote expert.
        metrics: Device selection status and fallback flag from the bounded allocator.
        valid_routes: Device validity flag from the original counts collective.
        entry_order: Stable local entry permutation prepared during counts communication.
        geometry: Device intervals compiled together with the allocation, if available.
        chunk_rows: Include exact round controls in the same Host export when provided.
        graphs: Shared cache for compiled round programs on NPU.

    Returns:
        Device maps and one small Host packet containing both complete and
        per-round controls. Bucket growth for unequal source extents is checked
        before a consumer can submit any collective.
    """
    ranks, experts = counts.shape
    home_count = experts // ranks
    device = ids.device
    counts, allocation = counts.to(device), allocation.to(device)
    capacity = min(experts - home_count, experts if replica_capacity is None else replica_capacity)
    if geometry is None:
        geometry = route_geometry(counts, allocation, rank, capacity)
    score = (torch.cat((candidate_metrics(allocation[None], counts.sum(0), experts)[0], counts.new_zeros(1)))
             if metrics is None else metrics)
    valid = ((counts >= 0).all() & (allocation >= 0).all()
             & (counts.sum(0) == allocation.sum(1)).all())
    if valid_routes is not None:
        valid = valid & valid_routes
    if metrics is not None:
        valid = valid & metrics[0].bool()
    # Wire order: six candidate metrics, fallback, route-valid, maximum receive
    # rows, four peer split tables, expert row counts, then optional chunk rows.
    controls = [score, valid.long().reshape(1), geometry.max_receive_rows.reshape(1),
                geometry.peer_rows[rank], geometry.peer_rows[:, rank],
                geometry.weight_send, geometry.weight_recv, geometry.groups, geometry.send_ids]
    rounds = None
    if chunk_rows is not None:
        rounds = compile_round_geometry(geometry, rank, chunk_rows, ids.numel(), graphs)
        controls.append(rounds.control.reshape(-1))
    # Include per-round controls in this packet. Large token maps remain on
    # device; no second steady-state Host wait is needed by chunk execution.
    pending = HostControl(torch.cat(controls))
    route_metadata = torch.cat((geometry.source_starts[rank]
                                - (counts[rank].cumsum(0, dtype=geometry.source_starts.dtype) - counts[rank]),
                                geometry.target_ends.reshape(-1)))
    dispatch_order = _entry_order(counts, allocation, rank, ids, route_metadata, entry_order)
    unpermute_indices = torch.empty(ids.numel(), dtype=torch.int32, device=device)
    unpermute_indices.scatter_(0, dispatch_order, torch.arange(ids.numel(), dtype=torch.int32, device=device))
    receive_counts, starts = None, None
    if build_receive_order:
        receive_counts = geometry.receive_rows.index_select(1, geometry.active)
        wire_starts = (geometry.receive_rows.reshape(-1).cumsum(0, dtype=geometry.receive_rows.dtype)
                       - geometry.receive_rows.reshape(-1)).reshape_as(geometry.receive_rows)
        starts = wire_starts.index_select(1, geometry.active).T
    boundaries = geometry.groups.cumsum(0, dtype=geometry.groups.dtype).long()
    # Submit all work independent of host splits before consuming the copy.
    # Its event precedes those maps, so this wait does not drain their stream.
    header = pending.tolist()
    (allocation_valid, moved_rows, peak_replica_count, replica_count, migration_lower_bound_rows,
     minimum_migration_certificate, planner_fallback, routes_valid, max_receive_rows) = header[:9]
    allocation_metrics = (allocation_valid, moved_rows, peak_replica_count, replica_count,
                          migration_lower_bound_rows, minimum_migration_certificate)
    if not routes_valid:
        raise RuntimeError(f'MoonEP invalid routing/allocation; metrics={[*allocation_metrics, planner_fallback]}')
    offset = 9
    token_send, token_recv, weight_send, weight_recv = (
        header[offset + index * ranks:offset + (index + 1) * ranks] for index in range(4))
    offset += 4 * ranks
    slots_capacity = geometry.active.numel()
    active_count = home_count + sum(weight_recv)
    group_counts = tuple(header[offset:offset + active_count])
    offset += slots_capacity
    send_slots = tuple(header[offset:offset + sum(weight_send)])
    offset += geometry.send_ids.numel()
    chunk_metadata = None
    if rounds is not None:
        count = (max_receive_rows + chunk_rows - 1) // chunk_rows
        if count > rounds.control.shape[0]:
            # Unequal source shapes can exceed the local sizing hint. Grow
            # before consuming any splits; subsequent calls reuse the bucket.
            rounds = compile_round_geometry(geometry, rank, chunk_rows, max_receive_rows, graphs)
            flat = HostControl(rounds.control).tolist()
            rows = tuple(tuple(row) for row in flat[:count])
        else:
            width = rounds.control.shape[1]
            rows = tuple(tuple(header[offset + index * width:offset + (index + 1) * width])
                         for index in range(count))
        chunk_metadata = ChunkMetadata(chunk_rows, rounds, rows)
    sort_order = None
    if build_receive_order:
        sort_order = expand_segments(starts[:active_count], receive_counts[:, :active_count].T, sum(token_recv))
    return ExecutionPlan(
        dispatch_order, unpermute_indices, sort_order,
        boundaries[:active_count], group_counts, token_send, token_recv, weight_send, weight_recv,
        geometry.send_ids[:sum(weight_send)], home_count, bool(replica_count), allocation_metrics,
        send_slots=send_slots, planner_fallback=bool(planner_fallback),
        chunk_metadata=chunk_metadata,
    )


def compile_host_plan(counts: torch.Tensor, allocation: torch.Tensor, ids: torch.Tensor,
                      rank: int, *, replica_capacity: int, entry_order: torch.Tensor | None = None,
                      planner_fallback: bool = False) -> ExecutionPlan:
    """Compile admitted CPU topology with one upload; keep all token maps on device.

    Args:
        counts: Validated CPU INT64 source/expert histogram from count admission.
        allocation: Feasible CPU expert/destination placement preserving those counts.
        ids: Original local expert choices on the compute device.
        rank: EP-local source and receiver rank.
        replica_capacity: Maximum number of remote expert slots.
        entry_order: Stable local choice order submitted while the count copy ran.
        planner_fallback: Whether the shared CPU allocator selected its fallback.

    Returns:
        The same NF execution contract as ``compile_plan``, including zero-row
        home slots. Only O(R*E) topology is decoded on the host. A single pinned
        transfer carries narrow interval metadata; O(M) row expansion, sorting
        and inverse maps remain on the device. No compiled program is created.
    """
    if counts.device.type != "cpu" or allocation.device.type != "cpu":
        raise ValueError("host geometry requires admitted CPU count and allocation tables")
    ranks, experts = counts.shape
    home_count = experts // ranks
    capacity = min(experts - home_count, replica_capacity)
    geometry = route_geometry(counts, allocation, rank, capacity)
    score = candidate_metrics(allocation[None], counts.sum(0), replica_capacity)[0]
    if not bool(score[0]):
        raise ValueError("invalid admitted expert placement")
    token_send = geometry.peer_rows[rank].tolist()
    token_recv = geometry.peer_rows[:, rank].tolist()
    weight_send, weight_recv = geometry.weight_send.tolist(), geometry.weight_recv.tolist()
    active_count = home_count + sum(weight_recv)
    group_counts = tuple(geometry.groups[:active_count].tolist())
    metadata = torch.cat((geometry.source_starts[rank] - (counts[rank].cumsum(0) - counts[rank]),
                          geometry.target_ends.reshape(-1)))
    receive = geometry.receive_rows.index_select(1, geometry.active)[:, :active_count].T
    flat = geometry.receive_rows.reshape(-1)
    wire_starts = (flat.cumsum(0) - flat).reshape_as(geometry.receive_rows)
    starts = wire_starts.index_select(1, geometry.active).T[:active_count]
    send_slots = tuple(geometry.send_ids[:sum(weight_send)].tolist())
    tensors = (metadata, geometry.groups[:active_count].cumsum(0),
               geometry.send_ids[:sum(weight_send)], starts.reshape(-1), receive.reshape(-1))
    lengths = [value.numel() for value in tensors]
    packet = torch.cat(tensors)
    if ids.device.type != "cpu":
        # Admission bounds each source to 2**20 rows, matching route_geometry's
        # INT32 NPU prefix path and avoiding AiCPU INT64 cumsum in row expansion.
        if ids.device.type == "npu" and ranks * 2**20 < 2**31:
            packet = packet.int()
        packet = packet.pin_memory().to(ids.device, non_blocking=True)
    metadata, boundaries, send_ids, starts, receive = packet.split(lengths)
    boundaries, send_ids = boundaries.long(), send_ids.long()
    dispatch_order = _entry_order(counts, allocation, rank, ids, metadata, entry_order)
    unpermute_indices = torch.empty(ids.numel(), dtype=torch.int32, device=ids.device)
    unpermute_indices.scatter_(0, dispatch_order, torch.arange(ids.numel(), dtype=torch.int32, device=ids.device))
    sort_order = expand_segments(starts, receive, sum(token_recv))
    return ExecutionPlan(
        dispatch_order, unpermute_indices, sort_order, boundaries, group_counts,
        token_send, token_recv, weight_send, weight_recv, send_ids, home_count,
        bool(score[3]), tuple(score.tolist()), send_slots=send_slots,
        planner_fallback=planner_fallback,
    )
