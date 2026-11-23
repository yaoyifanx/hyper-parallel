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
"""Fixed replica storage admitted before any weight or gradient collective."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch


def check_dense(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
                *, transposed: bool = False) -> None:
    """Validate a complete contiguous physical matrix storage, including views."""
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError("replica tensor does not match the admitted shape, dtype, or device")
    physical = value.transpose(-2, -1) if transposed else value
    if not physical.is_contiguous():
        raise ValueError("replica tensor does not match the admitted physical layout")
    capacity = value.untyped_storage().nbytes() // value.element_size()
    if value.storage_offset() < 0 or value.storage_offset() + value.numel() > capacity:
        raise ValueError("replica tensor exceeds its allocated storage")


def overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two validated dense tensors address overlapping bytes."""
    if not left.numel() or not right.numel():
        return False
    left_start, right_start = left.data_ptr(), right.data_ptr()
    return (left_start < right_start + right.numel() * right.element_size()
            and right_start < left_start + left.numel() * left.element_size())


def prepare_gradient_storage(allocator: Callable, shape: tuple[int, int], *,
                             staging_elements: tuple[int, int], incoming: torch.Tensor | None = None
                             ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Keep expert dV disjoint from early FP32 replica-gradient staging.

    Remote dW can become ready before home dW has finished reading dV. Send
    and receive storage therefore follows the entire BF16 input, with both
    FP32 view starts aligned to 32 bytes, including empty-token ranks.
    """
    if len(shape) != 2 or shape[0] < 0 or shape[1] < 1:
        raise ValueError("incoming gradient shape requires nonnegative rows and positive columns")
    if (len(staging_elements) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size < 0
                                        for size in staging_elements)):
        raise ValueError("gradient staging requires nonnegative send/receive element counts")
    send_elements, receive_elements = staging_elements
    if incoming is not None:
        check_dense(incoming, shape, torch.bfloat16, incoming.device)
        receive_start = ((send_elements + 7) // 8) * 8
        storage = allocator("gradient_staging", (receive_start + receive_elements,), torch.float32)
        check_dense(storage, (receive_start + receive_elements,), torch.float32, incoming.device)
        if storage.numel() and storage.data_ptr() % 32:
            raise ValueError("FP32 gradient staging must be aligned to 32 bytes")
        if overlaps(storage, incoming):
            raise ValueError("gradient staging must not overwrite the shared derivative")
        return incoming, (storage[:send_elements], storage[receive_start:receive_start + receive_elements])
    elements = prod(shape)
    send_start = ((elements + 1) // 2 + 7) // 8 * 8
    receive_start = (send_start + send_elements + 7) // 8 * 8
    capacity = 2 * (receive_start + receive_elements)
    storage = allocator("incoming", (capacity,), torch.bfloat16)
    check_dense(storage, (capacity,), torch.bfloat16, storage.device)
    if storage.data_ptr() % 32 or storage.storage_offset() % 2:
        raise ValueError("incoming gradient storage must support aligned BF16/FP32 reinterpretation")
    incoming = storage[:elements].view(shape)
    floating = storage.view(torch.float32)
    return incoming, (floating[send_start:send_start + send_elements],
                      floating[receive_start:receive_start + receive_elements])


@dataclass
class ReplicaSlot:
    """A fenced logical slot; initialized records successful use across retirement."""

    matrices: dict[str, torch.Tensor] = field(default_factory=dict)
    generation: int = 0
    in_use: bool = False
    last_read: Any = None
    initialized: bool = False


class ReplicaPool:
    """Own bounded BF16 replica slots; FP32 gradient domains retain geometry only."""

    def __init__(self, device: torch.device, dtype: torch.dtype, slot_count: int = 2) -> None:
        """Create an empty pool bound to one device and transfer dtype."""
        self.device, self.dtype = device, dtype
        if isinstance(slot_count, bool) or not isinstance(slot_count, int) or slot_count < 1:
            raise ValueError("replica slot count must be a positive integer")
        self.slot_count = slot_count
        self.home_count = 0
        self.replica_capacity = 0
        self.shapes: tuple[tuple[int, ...], ...] = ()
        self.transposed: tuple[bool, ...] = ()
        self.slots: list[ReplicaSlot] = []

    def allocate(self, home_count: int, replica_capacity: int, shapes: tuple[tuple[int, ...], tuple[int, ...]],
                 transposed: tuple[bool, bool]) -> None:
        """Set transfer geometry and create logical weight slots without tensor storage."""
        self._check_geometry(home_count, replica_capacity, shapes, transposed)
        self.home_count, self.replica_capacity = home_count, replica_capacity
        self.shapes, self.transposed = shapes, transposed
        if self.dtype == torch.bfloat16:
            self.slots = [ReplicaSlot() for _ in range(self.slot_count)]

    @staticmethod
    def _check_geometry(home_count: int, replica_capacity: int, shapes: tuple[tuple[int, ...], ...],
                        transposed: tuple[bool, ...]) -> None:
        if isinstance(home_count, bool) or not isinstance(home_count, int) or home_count < 1:
            raise ValueError("replica admission requires a positive home expert count")
        if isinstance(replica_capacity, bool) or not isinstance(replica_capacity, int) or replica_capacity < 0:
            raise ValueError("replica capacity must be a nonnegative integer")
        ReplicaPool._check_matrix_geometry(shapes, transposed)

    @staticmethod
    def _check_matrix_geometry(shapes: tuple[tuple[int, ...], ...], transposed: tuple[bool, ...]) -> None:
        """Check role geometry after validating the expert and replica capacities."""
        if len(shapes) != 2 or len(transposed) != 2 or any(not isinstance(value, bool) for value in transposed):
            raise ValueError("replica admission requires gate/down shapes and physical layouts")
        if any(len(shape) != 2 or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                                      for size in shape) for shape in shapes):
            raise ValueError("each replica matrix shape must contain two positive integers")

    def _allocate_matrices(self) -> dict[str, torch.Tensor]:
        """Build maximum-capacity matrices atomically, releasing partial failures."""
        matrices = {}
        try:
            for role, shape, transpose in zip(("gate", "down"), self.shapes, self.transposed):
                physical = tuple(reversed(shape)) if transpose else shape
                matrices[role] = torch.empty((self.replica_capacity, *physical), dtype=self.dtype,
                                            device=self.device)
                if transpose:
                    matrices[role] = matrices[role].transpose(-2, -1)
        except BaseException:
            matrices.clear()
            raise
        return matrices

    def ensure_storage(self, slot: ReplicaSlot, demand: int) -> None:
        """Materialize a borrowed slot at maximum capacity when replicas are needed.

        Unused logical slots retain no storage. The caller joins the last reader
        before reuse and converges allocation failures before submitting payloads.
        """
        if (isinstance(demand, bool) or not isinstance(demand, int)
                or not 0 <= demand <= self.replica_capacity):
            raise ValueError("replica demand must be within the admitted capacity")
        if slot.in_use:
            raise ValueError("replica storage can only be prepared for an idle slot")
        if demand == 0 or slot.matrices:
            return
        slot.matrices = self._allocate_matrices()
        slot.initialized = True

    def retire_slots(self) -> None:
        """Drop idle W storage references while preserving generations and fences."""
        if (self.dtype != torch.bfloat16 or len(self.slots) != self.slot_count
                or any(slot.in_use for slot in self.slots)):
            raise RuntimeError("weight retirement requires every admitted slot to be idle")
        for slot in self.slots:
            slot.matrices.clear()

    def restore_slots(self) -> None:
        """Rebuild retired W storage locally; the caller must admit it collectively."""
        if (self.dtype != torch.bfloat16 or len(self.slots) != self.slot_count
                or any(slot.in_use or slot.matrices for slot in self.slots)):
            raise RuntimeError("weight restoration requires every admitted slot to be empty and idle")
        try:
            for slot in self.slots:
                if slot.initialized:
                    slot.matrices = self._allocate_matrices()
        except BaseException:
            self.retire_slots()
            raise

    @property
    def reserved_bytes(self) -> int:
        """Return physical bytes without counting logical views twice."""
        return sum(value.numel() * value.element_size() for slot in self.slots for value in slot.matrices.values())

    def clear_unsubmitted(self) -> None:
        """Drop partial initialization allocations before any work is submitted."""
        for slot in self.slots:
            # Allocation failures retain frame locals in their traceback.
            slot.matrices.clear()
        self.slots.clear()
