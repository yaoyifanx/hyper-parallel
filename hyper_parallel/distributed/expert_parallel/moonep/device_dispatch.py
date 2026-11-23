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
"""NF GMM dispatch using the existing graph-free device-counted transport."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.routed_experts.descriptor import RouteDescriptor
from hyper_parallel.core.multicore.modules.routed_experts.transport import (
    PreparedExchange, prepare_exchange, requires_forward_groups, submit_prepared,
)
from hyper_parallel.core.multicore.modules.routed_experts.workspace import RingWorkspace, RingSpec
from hyper_parallel.distributed.expert_parallel.moonep.fused.plan import compile_local_route
from hyper_parallel.distributed.expert_parallel.moonep.fused.communicators import reserve_communicators
from hyper_parallel.distributed.expert_parallel.moonep.device_planning import expand_segments


@dataclass(frozen=True)
class NativeDispatchPlan:
    """Retain one invocation's native route and device-only row permutations."""
    route: RouteDescriptor
    source_rows: torch.Tensor
    target_rows: torch.Tensor
    target_identity: bool = False


def compile_device_forward(counts: torch.Tensor, allocation: torch.Tensor, rank: int, entry_order: torch.Tensor,
                           top_k: int) -> NativeDispatchPlan:
    """Map the shared CPU placement into native FD and NF home-first coordinates.

    Args:
        counts: Admitted CPU INT64 source/expert histogram.
        allocation: Feasible CPU placement preserving those counts.
        rank: Local rank within the EP group.
        entry_order: Device permutation in stable global-expert order.
        top_k: Original router fanout.

    Returns:
        Native range descriptors and device row maps. Only a small expert
        interval packet crosses H2D; no O(M) routing array is read by the host.
    """
    route = compile_local_route(counts, allocation, rank)
    home_count = counts.shape[1] // counts.shape[0]
    expert_ids = torch.arange(counts.shape[1], device=allocation.device)
    remote = expert_ids[(expert_ids // home_count != rank) & (allocation[:, rank] > 0)]
    active = torch.cat((expert_ids[rank * home_count:(rank + 1) * home_count], remote))
    rows = allocation[:, rank]
    starts = (rows.cumsum(0) - rows).index_select(0, active)
    lengths = rows.index_select(0, active)
    nonempty = active[lengths > 0]
    target_identity = bool(torch.equal(nonempty, torch.nonzero(rows > 0, as_tuple=False).flatten()))
    if target_identity:
        target_rows = torch.arange(route.received_rows, device=entry_order.device)
    else:
        metadata = torch.stack((starts, lengths)).int()
        if entry_order.device.type != 'cpu':
            metadata = metadata.pin_memory().to(entry_order.device, non_blocking=True)
        target_rows = expand_segments(metadata[0], metadata[1], route.received_rows)
    source_rows = (entry_order.int() // top_k).long()
    return NativeDispatchPlan(route, source_rows, target_rows, target_identity)


def prepare_device_forward(hidden: torch.Tensor, plan: NativeDispatchPlan, workspace: RingWorkspace, *, counted: bool,
                           ratio: float) -> PreparedExchange:
    """Converge local FD preparation failures before publishing native payloads."""
    error = None
    result = None
    try:
        metadata = torch.empty(
            (plan.route.received_rows, 8), device=hidden.device, dtype=torch.int64) if (
                counted and requires_forward_groups(0.0)) else None
        result = prepare_exchange(hidden.reshape(-1, hidden.shape[-1]).contiguous(),
                                  plan.route,
                                  workspace,
                                  mapping=plan.source_rows,
                                  dedup=counted,
                                  group_metadata=metadata,
                                  publish=False,
                                  forward_compact_max_unique_ratio=ratio,
                                  reduce_dense_ratio=0.0)
    except Exception as exc:
        error = exc
    workspace.check_preflight(error)
    return result


class DeviceDispatcher:
    """Own a model-wide FD window, preserving the caller's replica W/G stream."""

    def __init__(self) -> None:
        """Defer collective resources until the first NPU invocation."""
        self.workspace: RingWorkspace | None = None
        self.lease: Any = None
        self.closed = False
        self.failed = False

    def initialize(self, group: Any, ranks: int, hidden: torch.Tensor, stream: Any) -> None:
        """Admit matching resource leases before constructing the shared window."""
        if self.failed:
            raise RuntimeError('failed native FD initialization requires whole-job termination')
        if self.closed:
            raise RuntimeError('closed device dispatcher')
        if self.workspace is not None:
            if self.workspace.spec.hidden != hidden.shape[-1] or self.workspace.device != hidden.device:
                raise ValueError('all NF device dispatchers must share hidden width and device')
            return
        error = None
        manifest = None
        try:
            if hidden.device.type != 'npu' or hidden.dtype != torch.bfloat16:
                raise ValueError('selection_v1 NF dispatch requires NPU BF16 activations')
            self.lease = reserve_communicators(tuple(dist.get_process_group_ranks(group)))
            manifest = (ranks, hidden.shape[-1], self.lease.slot)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        records = [None] * ranks
        dist.all_gather_object(records, (manifest, error), group=group)
        if any((v[1] is not None or v[0] != manifest for v in records)):
            if self.lease is not None:
                self.lease.release()
                self.lease = None
            raise RuntimeError('NF device dispatch manifest mismatch or preparation failure')
        try:
            groups = self.lease.initialize()
            self.workspace = RingWorkspace(RingSpec(ranks, hidden.shape[-1]),
                                           group,
                                           control_group=groups[0],
                                           stream=stream)
        except BaseException:
            self.failed = True
            raise

    def forward(self, hidden: torch.Tensor, plan: NativeDispatchPlan, *, counted: bool, ratio: float) -> torch.Tensor:
        """Dispatch raw token rows, then restore the existing NF GMM expert order."""
        prepared = prepare_device_forward(hidden, plan, self.workspace, counted=counted, ratio=ratio)
        value = submit_prepared(prepared, self.workspace).wait()
        # The admitted CPU topology proves identity, including zero-row homes.
        # Keep the received storage when NF and native expert order coincide.
        return value if plan.target_identity else value.index_select(0, plan.target_rows)

    def forget(self, plan: NativeDispatchPlan) -> None:
        """Retire completed invocation metadata while preserving window sequences."""
        for phase in (False, True, 'readiness'):
            self.workspace.metadata.pop((plan.route, phase), None)
        self.workspace.metadata.pop((plan.route, 'host_transport', self.workspace.spec.block_rows), None)

    def close(self) -> None:
        """Collectively release the window after all model work has drained."""
        if self.failed:
            raise RuntimeError('failed native FD initialization requires whole-job termination')
        if self.closed:
            return
        if self.workspace is not None:
            self.workspace.close()
            self.workspace = None
        if self.lease is not None:
            self.lease.release()
            self.lease = None
        self.closed = True
