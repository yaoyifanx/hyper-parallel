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
"""Single packed replica exchanges; home weights always remain owned by FSDP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from math import prod
from threading import RLock
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore.modules.routed_experts.readiness import ProducerFence
from hyper_parallel.core.multicore.modules.routed_experts.ordered_sum import OrderedSumWork, prepare_ordered_sum
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import ReplicaPool, check_dense, overlaps
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import (
    Fragment, LocalReplicaTransfer, ReplicaTransfer, copy_matrix,
)


def _submission(method: Callable) -> Callable:
    """Serialize a complete domain submission and poison it on any failure."""
    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        with self._lock:  # pylint: disable=protected-access
            if self.poisoned:
                raise RuntimeError("replica communication domain is poisoned")
            try:
                return method(self, *args, **kwargs)
            except BaseException:
                self.poisoned = True
                raise
    return _wrapped


@dataclass
class WeightLease:
    """Remote views borrowed until release; home weights remain owned by FSDP."""

    remote: dict[tuple[int, str], torch.Tensor]
    ready: Any
    owner: Any
    slot: int | None
    generation: int
    released: bool = False


@dataclass
class GradientPayload:
    """Own one prepared replica return from remote readiness through owner reduction."""

    owner: Any
    gradients: tuple[torch.Tensor, torch.Tensor]
    outputs: tuple[torch.Tensor, torch.Tensor]
    buffers: tuple[torch.Tensor, torch.Tensor]
    transfer: LocalReplicaTransfer | None
    indices: dict[int, int]
    producer: ProducerFence
    reduction: OrderedSumWork | None
    ready: Any = None
    finished: bool = False
    accumulated: bool = False
    direct_send: bool = False


class TransferDomain:
    """Replica storage using the runtime's shared payload communicator and stream.

    The caller converges W/G admission and phase preparation across ranks. This
    domain performs no status collective during materialization or gradient return. Explicit
    restoration of retired W storage adds one caller-ordered admission.
    """

    def __init__(self, group: Any, rank: int, device: torch.device, *, dtype: torch.dtype = torch.bfloat16,
                 control_group: Any = None,
                 weight_slots: int = 2, stream: Any) -> None:
        """Borrow the common payload stream; call admit before submission."""
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("replica domains require BF16 weights or FP32 gradients")
        self.group, self.rank, self.device = group, rank, device
        self.control_group = group if control_group is None else control_group
        self.dtype = dtype
        self.stream = stream
        self.buffers: dict[str, torch.Tensor] = {}
        self.status = torch.zeros(1, dtype=torch.int32, device="cpu" if control_group is not None else device)
        self.admitted = False
        self.parked = False
        self.poisoned = False
        self._lock = RLock()
        self._ranks = dist.get_world_size(group)
        self._pool = ReplicaPool(device, dtype, slot_count=weight_slots)
        self._lease_owner = object()

    @torch.no_grad()
    @_submission
    def admit(self, home_count: int, replica_capacity: int, shapes: tuple[tuple[int, ...], tuple[int, ...]],
              transposed: tuple[bool, bool]) -> None:
        """Reserve replica storage during one caller preflight.

        This is local allocation only. The caller must converge success before any
        rank submits payloads. Failed partial allocations are dropped locally.
        Gradient domains retain geometry only and borrow their phase's exact staging.
        """
        if self.admitted:
            raise RuntimeError("replica communication domain has already been admitted")
        try:
            if not 0 <= self.rank < self._ranks:
                raise ValueError("replica rank is outside its communication group")
            self._pool.allocate(home_count, replica_capacity, shapes, transposed)
            # The first submission may arrive on a different compute stream.
            self.stream.wait_stream(torch.npu.current_stream(self.device))
        except BaseException:
            self.buffers.clear()
            self._pool.clear_unsubmitted()
            raise
        self.admitted = True

    @property
    def reserved_bytes(self) -> int:
        """Return persistent W/G storage bytes, excluding caller-owned gradients."""
        return self._pool.reserved_bytes + sum(value.numel() * value.element_size()
                                             for value in self.buffers.values())

    @_submission
    def retire_storage(self) -> int:
        """Drop idle W references without draining streams or invalidating views.

        Existing Tensor stream records and queued command references protect
        prior users. Slot generations and last-reader events remain unchanged.
        Returns bytes no longer owned here, not bytes already returned to HBM.
        """
        if not self.admitted or self.dtype != torch.bfloat16:
            raise RuntimeError("only an admitted weight domain can retire storage")
        if self.parked:
            return 0
        released = self.reserved_bytes
        self._pool.retire_slots()
        self.buffers.clear()
        self.parked = True
        return released

    @torch.no_grad()
    @_submission
    def restore_storage(self, error: Exception | None = None) -> None:
        """Rebuild retired W storage and converge once before any W payload.

        Every rank must call at the same admitted first sparse demand, including
        empty receivers. A local allocation or caller validation error is carried
        by the existing control group. Remote failure drops local new references.
        """
        if error is None:
            try:
                if not self.admitted or not self.parked or self.dtype != torch.bfloat16:
                    raise RuntimeError("weight restoration requires an admitted parked domain")
                self._pool.restore_slots()
                self.stream.wait_stream(torch.npu.current_stream(self.device))
            except Exception as exc:
                error = exc
        try:
            self.check_preflight(error)
        except BaseException:
            if self.parked:
                self.buffers.clear()
                self._pool.retire_slots()
            raise
        self.parked = False


    def check_preflight(self, error: Exception | None) -> None:
        """Converge preparation before the fixed-order replica collectives."""
        self.status.fill_(int(error is not None or self.poisoned))
        if self._ranks > 1:
            dist.all_reduce(self.status, op=dist.ReduceOp.MAX, group=self.control_group)
        if int(self.status.item()):
            self.poisoned = True
            raise RuntimeError("replica preparation failed; terminate this distributed execution") from error

    @torch.no_grad()
    def _exchange(self, transfer: LocalReplicaTransfer | None, source: Callable,
                  target: Callable | None = None, *,
                  staging: tuple[torch.Tensor, torch.Tensor] | None = None,
                  packed_ready: bool = False) -> None:
        """Pack, exchange, wait and unpack once on the shared EP stream.

        None means no migration anywhere in the group. A nonempty global
        descriptor still submits this collective on a rank with zero splits.

        Args:
            transfer: Cached local matrix views and peer splits in tensor elements.
            source: Callback returning the complete source matrix for (expert, role).
            target: Optional callback returning a weight destination; gradients stay packed.
            staging: Optional borrowed send/receive buffers; otherwise use the EP-owned pool.
            packed_ready: Producer wrote the exact peer-ordered send slices directly.
        """
        if transfer is None:
            return
        send_buffer, receive_buffer = self._staging(transfer) if staging is None else staging
        packed = send_buffer[:sum(transfer.send_sizes)]
        returned = receive_buffer[:sum(transfer.receive_sizes)]
        # Private W staging is allocated, packed and consumed on this EP
        # stream. Recording its allocation stream delays NPU cache reuse.
        # Borrowed gradient staging may originate on C and still needs it.
        if staging is not None:
            packed.record_stream(self.stream)
            returned.record_stream(self.stream)
        if not packed_ready:
            offset = 0
            for fragment in transfer.sent:
                copy_matrix(packed[offset:offset + fragment.elements], source(fragment.expert, fragment.role))
                offset += fragment.elements
        role = "gradient_replica" if self.dtype == torch.float32 else "weight_replica"
        with torch.profiler.record_function(f"moonep::all_to_all_v::{role}"):
            work = dist.all_to_all_single(returned, packed, list(transfer.receive_sizes), list(transfer.send_sizes),
                                          group=self.group, async_op=True)
        work.wait()
        if target is None:
            return
        offset = 0
        for fragment in transfer.received:
            copy_matrix(returned[offset:offset + fragment.elements], target(fragment.expert, fragment.role),
                        unpack=True)
            offset += fragment.elements

    @torch.no_grad()
    @_submission
    def materialize(self, active: tuple[int, ...], home_count: int, read_weights: Callable,
                    transfer: ReplicaTransfer | None, shapes: tuple[tuple[int, ...], tuple[int, ...]],
                    transposed: tuple[bool, bool]) -> WeightLease:
        """Borrow one admitted remote slot, then read current FSDP views on W."""
        self._check_call(active, home_count, torch.bfloat16)
        if shapes != self._pool.shapes or transposed != self._pool.transposed:
            raise ValueError("weight layout differs from replica admission")
        prepared = self._prepare_transfer(transfer, active)
        if self._pool.compact:
            error = None
            try:
                slot_index, generation, remote = self._claim_slot(active)
            except Exception as exc:
                error = exc
            self.check_preflight(error)
        else:
            slot_index, generation, remote = self._claim_slot(active)
        current = torch.npu.current_stream(self.device)
        if self.stream != current:
            self.stream.wait_stream(current)
        with torch.npu.stream(self.stream):
            gate_up, down = read_weights()
            for value, shape, transpose in zip((gate_up, down), shapes, transposed):
                check_dense(value, (home_count, *shape), torch.bfloat16, self.device, transposed=transpose)
            for value in (gate_up, down, *remote.values()):
                value.record_stream(self.stream)
            # Startup admission requires matching physical layouts on all ranks.
            # Copy their contiguous physical rows directly: slicing logical rows
            # of a transposed view can otherwise fall back to AiCPU ViewCopy.
            roles = ("gate", "down")
            weights = {role: value.transpose(-2, -1) if transpose else value
                       for role, value, transpose in zip(roles, (gate_up, down), transposed)}
            physical_remote = {
                (expert, role): value.transpose(-2, -1) if transposed[roles.index(role)] else value
                for (expert, role), value in remote.items()
            }
            self._exchange(prepared, lambda expert, role: weights[role][expert % home_count],
                           lambda expert, role: physical_remote[expert, role])
            ready = torch.npu.Event()
            ready.record(self.stream)
        return WeightLease(remote, ready, self._lease_owner, slot_index, generation)

    @_submission
    def release_lease(self, lease: WeightLease) -> None:
        """Expire a W borrow after its last use has been enqueued on current C.

        C waits for readiness even if a prefetch was cancelled without a reader.
        Reuse on W waits for the recorded C event. No later reads through this
        lease are valid after release, including reads on unrelated streams.
        """
        if lease.owner is not self._lease_owner or lease.released:
            raise ValueError("weight lease belongs to another domain or has already been released")
        current = torch.npu.current_stream(self.device)
        current.wait_event(lease.ready)
        if lease.slot is not None:
            slot = self._pool.slots[lease.slot]
            if not slot.in_use or slot.generation != lease.generation:
                raise ValueError("weight lease no longer owns its replica slot")
            slot.last_read = torch.npu.Event()
            slot.last_read.record(current)
            slot.in_use = False
        lease.released = True

    @torch.no_grad()
    def prepare_gradient_payload(self, active: tuple[int, ...], home_count: int,
                                 gradients: tuple[torch.Tensor, torch.Tensor], transfer: ReplicaTransfer | None,
                                 producer: ProducerFence, *, outputs: tuple[torch.Tensor, torch.Tensor],
                                 staging: tuple[torch.Tensor, torch.Tensor]) -> GradientPayload:
        """Bind immutable local/peer-ordered sources without reading unready dW.

        The same frame retains its exact communication storage and owner outputs
        through submission and completion; no runtime-wide per-role maps are needed.
        """
        self._check_call(active, home_count, torch.float32)
        prepared = self._prepare_transfer(transfer, active, reverse=True)
        if len(gradients) != 2 or len(outputs) != 2 or staging is None:
            raise ValueError("gradient frames require gate/down tensors and caller staging")
        buffers = self._staging(prepared, staging)
        for gradient, output, shape in zip(gradients, outputs, self._pool.shapes):
            check_dense(gradient, (len(active), *shape), torch.float32, self.device)
            check_dense(output, (home_count, *shape), torch.bfloat16, self.device)
        if (any(overlaps(value, buffer) for value in (*gradients, *outputs) for buffer in buffers)
                or any(overlaps(output, gradient) for output in outputs for gradient in gradients)
                or overlaps(*outputs)):
            raise ValueError("gradient payloads, outputs and writable communication storage must not alias")
        indices = {expert: index for index, expert in enumerate(active)}
        groups, targets, local = [], [], []
        roles = ("gate", "down")
        for gradient, output in zip(gradients, outputs):
            for slot in range(home_count):
                expert = self.rank * home_count + slot
                present = expert in indices
                groups.append([gradient[indices[expert]]] if present else [])
                targets.append(output[slot])
                local.append(present)
        if prepared is not None:
            offset = 0
            for fragment in prepared.received:
                group = roles.index(fragment.role) * home_count + fragment.expert % home_count
                groups[group].append(buffers[1][offset:offset + fragment.elements].view(targets[group].shape))
                offset += fragment.elements
        reduction = prepare_ordered_sum(groups, targets, local)
        return GradientPayload(self, gradients, outputs, buffers, prepared, indices, producer, reduction)

    def accumulation_targets(self, active: tuple[int, ...], home_count: int, outputs: tuple,
                             transfer: ReplicaTransfer | None, *, staging: tuple) -> tuple:
        """Map remote FP32 producers to exact send slices and homes to their real g.

        The caller retains the disjoint dY, send, and receive storage until all
        producers and communication readers finish. No gradient values are read.
        """
        self._check_call(active, home_count, torch.float32)
        prepared = self._prepare_transfer(transfer, active, reverse=True)
        buffers = self._staging(prepared, staging)
        return self._accumulation_targets(active, home_count, outputs, prepared, buffers)

    def _accumulation_targets(self, active: tuple[int, ...], home_count: int, outputs: tuple,
                              prepared: LocalReplicaTransfer | None, buffers: tuple) -> tuple:
        self._check_accumulation_outputs(home_count, outputs, buffers)
        remote, offset = {}, 0
        if prepared is not None:
            for fragment in prepared.sent:
                shape = outputs[fragment.role == "down"].shape[1:]
                remote[fragment.expert, fragment.role] = buffers[0][offset:offset + fragment.elements].view(shape)
                offset += fragment.elements
        return tuple(tuple(output[expert % home_count] if expert // home_count == self.rank
                           else remote[expert, role] for expert in active)
                     for role, output in zip(("gate", "down"), outputs))

    def _check_accumulation_outputs(self, home_count: int, outputs: tuple, buffers: tuple) -> None:
        if len(outputs) != 2:
            raise ValueError("accumulated gradients require both owner matrices")
        for output, shape, transpose in zip(outputs, self._pool.shapes, self._pool.transposed):
            physical = tuple(reversed(shape)) if transpose else shape
            check_dense(output, (home_count, *physical), torch.float32, self.device)
            if any(overlaps(output, buffer) for buffer in buffers):
                raise ValueError("owner gradients must not alias replica communication storage")
        if overlaps(*outputs):
            raise ValueError("owner gate and down gradients must not alias")

    def prepare_accumulated_payload(self, active: tuple[int, ...], home_count: int,
                                    gradients: tuple, transfer: ReplicaTransfer | None,
                                    producer: ProducerFence, *, outputs: tuple,
                                    staging: tuple, direct_send: bool = False) -> GradientPayload:
        """Keep home FP32 g local; send only independent FP32 replica contributions.

        All matrices use the owner's contiguous physical parameter layout. Metadata
        carries element counts, so the exchange does not transpose or cast payloads.
        """
        self._check_call(active, home_count, torch.float32)
        prepared = self._prepare_transfer(transfer, active, reverse=True)
        buffers = self._staging(prepared, staging)
        remote_addresses = {}
        if direct_send:
            self._check_accumulation_outputs(home_count, outputs, buffers)
            offset = 0
            if prepared is not None:
                for fragment in prepared.sent:
                    remote_addresses[fragment.expert, fragment.role == "down"] = buffers[0].data_ptr() + offset * 4
                    offset += fragment.elements
        for role in range(2):
            if len(gradients[role]) != len(active) or outputs[role].dtype != torch.float32:
                raise ValueError("accumulated gradients must cover each active expert in FP32")
            shape, home_address = outputs[role].shape[1:], outputs[role].data_ptr()
            matrix_bytes = prod(shape) * 4
            for index, value in enumerate(gradients[role]):
                if value.dtype != torch.float32 or not value.is_contiguous():
                    raise ValueError("mapped gradients require contiguous physical FP32 matrices")
                if direct_send:
                    expert = active[index]
                    address = (home_address + (expert % home_count) * matrix_bytes
                               if expert // home_count == self.rank else remote_addresses[expert, role])
                    if value.device != outputs[role].device or value.shape != shape or value.data_ptr() != address:
                        raise ValueError("direct gradients must match their exact owner or peer-ordered send slice")
                elif any(overlaps(value, buffer) for buffer in buffers):
                    raise ValueError("mapped gradients must not alias communication staging")
        indices = {expert: index for index, expert in enumerate(active)}
        return GradientPayload(self, gradients, outputs, buffers, prepared, indices, producer, None,
                               accumulated=True, direct_send=direct_send)

    @torch.no_grad()
    @_submission
    def submit_gradient_payload(self, packet: GradientPayload) -> None:
        """Exchange remote dW while C continues dX and home dW production."""
        if packet.owner is not self or packet.ready is not None or packet.finished:
            raise ValueError("gradient payload is foreign or has already been submitted")
        if packet.transfer is None:
            # No remote reader exists. C already orders the complete tail before
            # owner reduction, so an empty G submission needs no producer poll.
            packet.ready = torch.npu.Event()
            packet.ready.record(torch.npu.current_stream(self.device))
            return
        with torch.npu.stream(self.stream):
            packet.producer.wait_on_stream(self.stream)
            # Home g is never sent on G. Recording it there would unnecessarily
            # turn the full accumulator into cross-stream allocator storage.
            gradients = (tuple(packet.gradients[fragment.role == "down"][packet.indices[fragment.expert]]
                               for fragment in packet.transfer.sent)
                         if packet.accumulated else packet.gradients)
            for value in (*gradients, *packet.buffers):
                value.record_stream(self.stream)
            self._exchange(
                packet.transfer, lambda expert, role: packet.gradients[role == "down"][packet.indices[expert]],
                staging=packet.buffers, packed_ready=packet.direct_send)
            packet.ready = torch.npu.Event()
            packet.ready.record(self.stream)

    @torch.no_grad()
    @_submission
    def complete_gradients(self, packet: GradientPayload) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge on C after home dW producers, then return both complete owner gradients."""
        if packet.owner is not self or packet.finished or packet.ready is None:
            raise ValueError("gradient payload is foreign, unsubmitted or has already been completed")
        current = torch.npu.current_stream(self.device)
        current.wait_event(packet.ready)
        for value in packet.buffers:
            value.record_stream(current)
        with torch.profiler.record_function("moonep::owner_reduce"):
            if packet.accumulated:
                if packet.transfer is not None:
                    offset = 0
                    for fragment in packet.transfer.received:
                        role = fragment.role == "down"
                        target = packet.outputs[role][fragment.expert % packet.outputs[role].shape[0]]
                        target.add_(packet.buffers[1][offset:offset + fragment.elements].view_as(target))
                        offset += fragment.elements
            else:
                packet.reduction.run()
        packet.finished = True
        return packet.outputs

    def _staging(self, transfer: LocalReplicaTransfer | None,
                 staging: tuple[torch.Tensor, torch.Tensor] | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """Borrow exact phase views or grow reusable buffers to actual fanout sizes."""
        sizes = ((sum(transfer.send_sizes), sum(transfer.receive_sizes)) if transfer is not None else (0, 0))
        if staging is None:
            for name, elements in zip(("send", "receive"), sizes):
                buffer = self.buffers.get(name)
                if buffer is None or buffer.numel() < elements:
                    self.buffers[name] = torch.empty(elements, dtype=self.dtype, device=self.device)
            return self.buffers["send"], self.buffers["receive"]
        if not isinstance(staging, tuple) or len(staging) != 2:
            raise ValueError("gradient staging requires a send/receive tensor pair")
        for value, elements in zip(staging, sizes):
            check_dense(value, (elements,), self.dtype, self.device)
            if value.numel() and value.data_ptr() % 32:
                raise ValueError("gradient staging must have 32-byte aligned addresses")
        if overlaps(*staging):
            raise ValueError("gradient send and receive staging must not alias")
        return staging

    def _check_call(self, active: tuple[int, ...], home_count: int, dtype: torch.dtype) -> None:
        if not self.admitted or self.parked or self.dtype != dtype or home_count != self._pool.home_count:
            raise ValueError("replica domain admission does not match this operation")
        if (len(set(active)) != len(active)
                or any(isinstance(expert, bool) or not isinstance(expert, int)
                       or not 0 <= expert < home_count * self._ranks for expert in active)):
            raise ValueError("active expert ids must be unique and within the admitted ownership range")

    def _claim_slot(self, active: tuple[int, ...]) -> tuple[int | None, int, dict[tuple[int, str], torch.Tensor]]:
        experts = tuple(expert for expert in active if expert // self._pool.home_count != self.rank)
        if len(experts) > self._pool.replica_capacity:
            raise ValueError("active remote experts exceed the admitted replica capacity")
        # Empty receivers still consume a lease so slot availability follows
        # the same collective submission schedule on every rank.
        for index, slot in enumerate(self._pool.slots):
            if not slot.in_use:
                if slot.last_read is not None:
                    self.stream.wait_event(slot.last_read)
                if self._pool.compact:
                    with torch.npu.stream(self.stream):
                        self._pool.ensure_capacity(slot, len(experts))
                slot.generation += 1
                slot.in_use = True
                remote = {(expert, role): value[row] for row, expert in enumerate(experts)
                          for role, value in slot.matrices.items()}
                return index, slot.generation, remote
        raise RuntimeError("all replica slots are leased; skip prefetch until a lease is released")

    def _prepare_transfer(self, transfer: ReplicaTransfer | None, active: tuple[int, ...], *,
                          reverse: bool = False) -> LocalReplicaTransfer | None:
        """Validate one global descriptor, including remote coverage for None.

        Args:
            transfer: Global weight or gradient exchange; None means no group migration.
            active: Global expert IDs whose token rows are computed on this rank.
            reverse: Validate replica-to-owner gradient traffic instead of owner-to-replica weights.

        Returns:
            The descriptor's cached local view, or None after verifying no remote expert is required.
        """
        local, selected = None, set()
        if transfer is not None:
            if not isinstance(transfer, ReplicaTransfer) or transfer.ranks != self._ranks:
                raise ValueError("replica transfer does not match the admitted communication group")
            local = transfer.local(self.rank)
            self._check_peer_order(local)
            covered = set()
            for fragment in (*local.sent, *local.received):
                self._check_fragment(fragment, reverse)
                key = (fragment.source, fragment.target, fragment.expert, fragment.role)
                if key in covered:
                    raise ValueError("replica messages must contain each matrix exactly once")
                covered.add(key)
            selected = {(fragment.expert, fragment.role) for fragment in
                        (local.sent if reverse else local.received)}
        expected = {(expert, role) for expert in active
                    if expert // self._pool.home_count != self.rank for role in ("gate", "down")}
        if selected != expected:
            raise ValueError("replica transfer does not completely cover the local remote experts")
        return local

    @staticmethod
    def _check_peer_order(local: LocalReplicaTransfer) -> None:
        sent = tuple(fragment.target for fragment in local.sent)
        received = tuple(fragment.source for fragment in local.received)
        if sent != tuple(sorted(sent)) or received != tuple(sorted(received)):
            raise ValueError("replica fragments must follow the all-to-all peer order")

    def _role_elements(self, role: str) -> int:
        if role not in ("gate", "down"):
            raise ValueError("replica transfer role must be gate or down")
        return prod(self._pool.shapes[0 if role == "gate" else 1])

    def _check_fragment(self, fragment: Fragment, reverse: bool) -> None:
        indices = (fragment.source, fragment.target, fragment.expert, fragment.elements)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in indices):
            raise ValueError("replica fragment indices and counts must be integers")
        if (not 0 <= fragment.source < self._ranks or not 0 <= fragment.target < self._ranks
                or not 0 <= fragment.expert < self._pool.home_count * self._ranks):
            raise ValueError("replica fragment exceeds the admitted peer/expert range")
        owner = fragment.expert // self._pool.home_count
        direction_owner = fragment.target if reverse else fragment.source
        if direction_owner != owner or fragment.source == fragment.target:
            raise ValueError("replica fragment does not match ownership or gradient role")
        if fragment.elements != self._role_elements(fragment.role):
            raise ValueError("replica fragment exceeds its admitted matrix bounds")

    def drain(self) -> None:
        """Wait for all local readers before a normal shutdown or owner reshard."""
        torch.npu.current_stream(self.device).wait_stream(self.stream)
