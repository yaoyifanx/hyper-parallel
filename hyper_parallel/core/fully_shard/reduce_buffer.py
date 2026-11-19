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
"""Bounded rank-major RS storage shared by one ordered reduction pipeline."""
from dataclasses import dataclass

import torch


@dataclass(eq=False)
class _Slot:
    tensor: torch.Tensor
    busy: bool = False
    ready: object = None


class ReduceBufferPool:
    """Lease final RS inputs until their collective has been joined.

    Capacity is a byte budget, not a per-parameter allocation. When every suitable
    slot is leased, callers use an uncached allocation instead of overwriting an
    input still read by HCCL. Idle slots may be evicted to accommodate new shapes.
    """

    def __init__(self, budget_bytes: int) -> None:
        if budget_bytes < 0:
            raise ValueError("RS buffer budget must be nonnegative")
        self.budget_bytes = budget_bytes
        self.slots = []
        self.hits = 0
        self.misses = 0
        self.fallbacks = 0

    @property
    def reserved_bytes(self) -> int:
        """Return retained storage bytes, counting each slot once."""
        return sum(slot.tensor.numel() * slot.tensor.element_size() for slot in self.slots)

    def acquire(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, object]:
        """Return an exclusively writable tensor and an optional lease token."""
        elements = 1
        for length in shape:
            elements *= length
        size = elements * dtype.itemsize
        for slot in sorted(self.slots, key=lambda value: value.tensor.numel()):
            value = slot.tensor
            if not slot.busy and value.device == device and value.dtype == dtype and value.numel() >= elements:
                if slot.ready is not None:
                    torch.get_device_module(device).current_stream(device).wait_event(slot.ready)
                slot.busy = True
                self.hits += 1
                return value[:elements].view(shape), slot
        self.misses += 1
        if size <= self.budget_bytes:
            for slot in list(self.slots):
                if self.reserved_bytes + size <= self.budget_bytes:
                    break
                if not slot.busy:
                    self.slots.remove(slot)
            if self.reserved_bytes + size <= self.budget_bytes:
                slot = _Slot(torch.empty(elements, dtype=dtype, device=device), True)
                self.slots.append(slot)
                return slot.tensor.view(shape), slot
        self.fallbacks += 1
        return torch.empty(shape, dtype=dtype, device=device), None

    def release(self, token: object) -> None:
        """Retire a lease after Work.wait has ordered completion on this stream."""
        if token is None:
            return
        if not any(slot is token for slot in self.slots) or not token.busy:
            raise ValueError("RS input lease is foreign or already released")
        if token.tensor.device.type != "cpu":
            device = token.tensor.device
            stream = torch.get_device_module(device).current_stream(device)
            token.tensor.record_stream(stream)
            token.ready = torch.get_device_module(device).Event()
            token.ready.record(stream)
        token.busy = False
