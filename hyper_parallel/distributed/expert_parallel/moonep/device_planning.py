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
"""Tensor-only small-table planning; token rows never pass through the host."""

from __future__ import annotations

from typing import Callable

import torch
from torch.fx.experimental.proxy_tensor import make_fx


def inverse_permutation(order: torch.Tensor) -> torch.Tensor:
    """Invert a permutation with the native NPU scatter-update semantic operator.

    Args:
        order: INT32 or INT64 bijection of ``[0, order.numel())``. Its producer
            owns validation; no device values are downloaded here.

    Returns:
        The inverse on the same device and with the original integer dtype.
        NPU maps whose extent fits INT32 use vector ScatterNdUpdate with
        narrow coordinates. The payload retains the consumer's integer dtype.
    """
    if order.ndim != 1 or order.dtype not in (torch.int32, torch.int64):
        raise ValueError('permutation must be a one-dimensional INT32 or INT64 tensor')
    inverse = torch.empty_like(order)
    if order.numel() == 0:
        return inverse
    positions = torch.arange(order.numel(), dtype=order.dtype, device=order.device)
    if order.device.type == 'npu' and order.numel() <= torch.iinfo(torch.int32).max:
        # TorchNPU remains optional for CPU/GPU users of the planning helpers.
        import torch_npu  # pylint: disable=import-outside-toplevel

        return torch_npu.npu_scatter_nd_update_(inverse, order.int().unsqueeze(1), positions)
    return inverse.scatter_(0, order.long(), positions)


def _descending_positions(values: torch.Tensor) -> torch.Tensor:
    """Return each small-axis element's exact stable rank."""
    width = values.shape[-1]
    flat = values.reshape(-1, width)
    indices = torch.arange(width, device=values.device)
    higher = (flat[:, None, :] > flat[:, :, None]).int()
    tied = (flat[:, None, :] == flat[:, :, None]).int()
    before = higher + tied * (indices[None] < indices[:, None]).int()
    return before.sum(2, dtype=torch.int32).reshape_as(values)


def _invert_small_axis_permutation(order: torch.Tensor) -> torch.Tensor:
    """Invert small permutations with a reduction instead of INT64 ScatterElements."""
    width = order.shape[-1]
    flat = order.reshape(-1, width)
    indices = torch.arange(width, device=order.device, dtype=order.dtype)
    result = ((flat[:, :, None] == indices) * indices.int()[:, None]).sum(1, dtype=torch.int32)
    return result.reshape_as(order).long()


def stable_descending(values: torch.Tensor) -> torch.Tensor:
    """Sort a small axis with deterministic vector comparisons and no AiCPU scatter."""
    return _invert_small_axis_permutation(_descending_positions(values))


def candidate_metrics(allocation: torch.Tensor, totals: torch.Tensor, capacity: int) -> torch.Tensor:
    """Return [valid, moved, peak, copies, lower_bound, certificate] for each candidate."""
    batches, experts, ranks = allocation.shape
    homes = torch.arange(experts, device=allocation.device) // (experts // ranks)
    rank_ids = torch.arange(ranks, device=allocation.device)
    remote = (homes[:, None] != rank_ids[None]).to(allocation.dtype)
    enabled = (allocation > 0).to(allocation.dtype) * remote
    incoming = enabled.sum(1, dtype=allocation.dtype)
    outgoing = enabled.reshape(batches, ranks, experts // ranks, ranks).sum((2, 3), dtype=allocation.dtype)
    total = totals.sum(dtype=totals.dtype)
    targets = total // ranks + (rank_ids.to(totals.dtype) < total % ranks)
    valid = ((allocation >= 0).int().reshape(batches, -1).sum(1, dtype=torch.int32) == experts * ranks).int()
    valid = valid * ((totals >= 0).int().sum(dtype=torch.int32) == experts).int()
    conserved = (allocation.sum(2, dtype=allocation.dtype) == totals).int()
    balanced = (allocation.sum(1, dtype=allocation.dtype) == targets).int()
    bounded = (incoming <= capacity).int()
    valid = valid * (conserved.sum(1, dtype=torch.int32) == experts).int()
    valid = valid * (balanced.sum(1, dtype=torch.int32) == ranks).int()
    valid = valid * (bounded.sum(1, dtype=torch.int32) == ranks).int()
    moved = (allocation * remote).sum((1, 2), dtype=allocation.dtype)
    peak_dtype = torch.float32 if experts * ranks < 2**24 else torch.int64
    peak = torch.maximum(incoming.to(peak_dtype).max(1).values,
                         outgoing.to(peak_dtype).max(1).values).to(totals.dtype)
    lower = (totals.reshape(ranks, -1).sum(1, dtype=totals.dtype) - targets).clamp_min(0)
    lower = lower.sum(dtype=totals.dtype).expand(batches)
    return torch.stack((valid.long(), moved, peak, enabled.sum((1, 2), dtype=allocation.dtype), lower,
                        (valid * (moved == lower).int()).long()), 1)


def allocation_candidates(totals: torch.Tensor, ranks: int) -> torch.Tensor:
    """Batch both bounded policies, preserving first-rank and stable-sort ties.

    Only the R-1 load updates are sequential. Expert/target interval intersections
    and both policies share tensor operations rather than per-expert launches.
    """
    experts = totals.numel()
    home_count = experts // ranks
    rank_ids = torch.arange(ranks, device=totals.device, dtype=totals.dtype)
    total = totals.sum(dtype=totals.dtype)
    targets = total // ranks + (rank_ids < total % ranks)
    home = totals.reshape(ranks, home_count)
    balance = (home.sum(1, dtype=totals.dtype) - targets).expand(2, ranks).clone()
    quotas = totals.new_zeros((2, ranks, ranks))
    fill = torch.arange(2, device=totals.device, dtype=totals.dtype)
    for _ in range(ranks - 1):
        source_index = balance.argmax(1)[:, None]
        target_index = balance.argmin(1)[:, None]
        surplus = balance.gather(1, source_index).squeeze(1)
        deficit = -balance.gather(1, target_index).squeeze(1)
        source = (rank_ids[None] == source_index.to(rank_ids.dtype)).to(totals.dtype)
        target = (rank_ids[None] == target_index.to(rank_ids.dtype)).to(totals.dtype)
        amount = fill * deficit + (1 - fill) * torch.minimum(surplus, deficit)
        quotas = quotas + source[:, :, None] * target[:, None] * amount[:, None, None]
        balance = balance + (target - source) * amount[:, None]
    # Match the existing CPU candidate's FP32 stable ordering, including ties.
    expert_order = stable_descending(home.float())
    ordered_experts = home.gather(1, expert_order)
    expert_ends = ordered_experts.cumsum(1, dtype=totals.dtype)[None, :, :, None]
    target_order = stable_descending(quotas.float())
    ordered_quotas = quotas.gather(2, target_order)
    target_ends = ordered_quotas.cumsum(2, dtype=totals.dtype)[:, :, None, :]
    moved = (torch.minimum(expert_ends, target_ends)
             - torch.maximum(expert_ends - ordered_experts[None, :, :, None],
                             target_ends - ordered_quotas[:, :, None, :])).clamp_min(0)
    by_expert = moved.gather(3, _invert_small_axis_permutation(target_order)[:, :, None].expand_as(moved))
    result = by_expert.gather(2, _invert_small_axis_permutation(expert_order)[None, :, :, None].expand_as(by_expert))
    remaining = home[None] - result.sum(3, dtype=totals.dtype)
    diagonal = rank_ids[:, None] == rank_ids[None]
    result = result + diagonal[None, :, None] * remaining[..., None]
    return result.reshape(2, experts, ranks)


def select_device_allocation(counts: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select on device; defer validity/error handling to the final host control packet."""
    ranks, experts = counts.shape
    # Histogram admission bounds each source by 2**20. This is a numerical
    # dtype choice, not an EP support limit; larger domains retain INT64.
    counter_dtype = torch.int32 if counts.device.type != 'cpu' and ranks * 2**20 < 2**31 else torch.int64
    totals = counts.sum(0, dtype=counter_dtype)
    if ranks == 1:
        allocation = totals[:, None]
        return allocation.long(), torch.cat((candidate_metrics(allocation[None], totals, capacity)[0],
                                             totals.new_zeros(1))).long()
    candidates = allocation_candidates(totals, ranks)
    metrics = candidate_metrics(candidates, totals, capacity)
    use_first = metrics[0, 5].bool()
    selected = torch.where(use_first, candidates[0], candidates[0 if ranks == 2 else 1])
    score = torch.where(use_first, metrics[0], metrics[0 if ranks == 2 else 1])
    return selected.long(), torch.cat((score, (~use_first).long().reshape(1))).long()


def peer_round_quotas(traffic: torch.Tensor, rounds: int,
                      valid_rounds: torch.Tensor | None = None) -> torch.Tensor:
    """Partition peer traffic with vectorized sender costs and deterministic receiver phases.

    The receiver loop is stateful. Each iteration replaces both old sender
    loops with prefix-window gathers and one broadcast remainder update.
    """
    ranks = traffic.shape[0]
    if rounds == 0:
        return traffic.new_empty((0, ranks, ranks))
    effective = rounds if valid_rounds is None else valid_rounds.clamp(1, rounds)
    quotient, remainder = traffic // effective, traffic % effective
    # These costs count only remainder rows. At this proven bound FP32 sums
    # are exact integers and use the native vector reduction/argmin path.
    cost_dtype = torch.float32 if ranks * ranks * rounds < 2**24 else torch.int64
    prefix_dtype = torch.int32 if ranks * rounds < 2**31 else torch.int64
    loads = torch.zeros((ranks, rounds), device=traffic.device, dtype=cost_dtype)
    extras = []
    receiver_order = stable_descending(remainder.sum(0))
    phases = torch.arange(rounds, device=traffic.device)
    keep = (None if valid_rounds is None else (phases < valid_rounds).to(cost_dtype))
    extended = (None if valid_rounds is None else torch.arange(2 * rounds, device=traffic.device) % effective)
    for ordinal in range(ranks):
        receiver = receiver_order[ordinal:ordinal + 1]
        # Each remainder is below rounds. An explicit accumulation dtype
        # prevents integer promotion to the INT64 AiCPU Cumsum implementation.
        counts = remainder.index_select(1, receiver).squeeze(1).to(prefix_dtype)
        starts = counts.cumsum(0, dtype=prefix_dtype) - counts
        circular = loads.repeat(1, 2) if extended is None else loads.index_select(1, extended)
        prefix = torch.cat((loads.new_zeros((ranks, 1)), circular.cumsum(1)), 1)
        first = (starts[:, None] + phases[None]) % effective
        costs = (prefix.gather(1, first + counts[:, None]) - prefix.gather(1, first)).sum(0)
        if keep is not None:
            costs = costs + (1 - keep) * (ranks * ranks * rounds)
        phase = costs.argmin()
        extra = ((phases[None] - starts[:, None] - phase) % effective < counts[:, None]).to(cost_dtype)
        if keep is not None:
            extra = extra * keep[None]
        loads = loads + extra
        extras.append(extra)
    extra_rows = torch.stack(extras).index_select(0, _invert_small_axis_permutation(receiver_order))
    result = quotient[None] + extra_rows.permute(2, 1, 0).to(traffic.dtype)
    return result if keep is None else result * keep.to(traffic.dtype)[:, None, None]


def expand_segments(starts: torch.Tensor, counts: torch.Tensor, total: int) -> torch.Tensor:
    """Expand interval starts/counts to one 1-D row map without data-dependent shapes."""
    counts = counts.reshape(-1)
    offsets = starts.reshape(-1) - (counts.cumsum(0, dtype=counts.dtype) - counts)
    return (torch.repeat_interleave(offsets, counts, output_size=total)
            + torch.arange(total, device=counts.device))


class HostControl:
    """Start a small D2H before submitting work independent of its host values.

    The copy and its completion event use the producer's current stream. Waiting
    for that event does not wait for device maps submitted after the event.
    """

    def __init__(self, values: torch.Tensor) -> None:
        """Enqueue a pinned copy while retaining both buffers until consumption."""
        self.source = values
        self.ready = None
        self.host = values
        if values.device.type != 'cpu':
            self.host = torch.empty_like(values, device='cpu', pin_memory=True)
            self.host.copy_(values, non_blocking=True)
            module = torch.get_device_module(values.device)
            # This fence gates a host read. Untimed NPU events have exposed
            # unwritten control buffers under sustained asynchronous submission.
            self.ready = module.Event(enable_timing=True)
            self.ready.record(module.current_stream(values.device))

    def wait(self) -> torch.Tensor:
        """Return the host tensor after its copy, without waiting for later device work."""
        if self.ready is not None:
            self.ready.synchronize()
            self.ready = None
        return self.host

    def tolist(self) -> list:
        """Wait only for the control copy, then decode it for collective submission."""
        return self.wait().tolist()


class TableGraph:
    """Compile small CANN tensor programs, retaining separately owned plan outputs."""

    def __init__(self, function: Callable, example: torch.Tensor, *, capacity: int = 0) -> None:
        """Bind one topology program; CPU callers need neither NPU nor compiler imports."""
        self.function = function
        self.device = example.device
        self.capacity = capacity
        self.compiled = None

    def __call__(self, value: torch.Tensor, *parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Fuse vector expressions instead of merely replaying hundreds of eager tasks."""
        if value.device != self.device:
            raise ValueError('table compiler inputs must stay on its bound device')
        if value.device.type != 'npu':
            result = self.function(value, *parameters)
            return (result,) if isinstance(result, torch.Tensor) else result
        if self.compiled is None:
            # TorchAir is an optional Ascend dependency; CPU and CUDA imports
            # must not initialize its NPU registrations or compiler runtime.
            from torch_npu.dynamo import torchair  # pylint: disable=C0415

            # Pure tensor programs need no Dynamo frame/guard cache. Trace a
            # fixed-capacity FX graph; actual round counts stay tensor inputs.
            def tensor_outputs(*arguments: torch.Tensor) -> tuple[torch.Tensor, ...]:
                """Normalize topology-program results to the tuple expected by the compiler."""
                result = self.function(*arguments)
                return (result,) if isinstance(result, torch.Tensor) else result

            graph = make_fx(tensor_outputs, tracing_mode='fake')(value, *parameters)
            self.compiled = torchair.get_npu_backend()(graph, [value, *parameters])
        result = self.compiled(value, *parameters)
        outputs = (result,) if isinstance(result, torch.Tensor) else result
        # Retained plans outlive subsequent invocations of the same compiler.
        return tuple(output.clone() for output in outputs)
