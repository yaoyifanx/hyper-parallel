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
"""Prepared ordered FP32 reductions with one final BF16 store per output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from hyper_parallel.core.multicore.modules.routed_experts.transport import _load_native


@dataclass
class OrderedSumWork:
    """Retain immutable source addresses until their ordered reduction is submitted."""

    sources: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]
    metadata: torch.Tensor
    initialized: Any
    submitted: bool = False

    @torch.no_grad()
    def run(self) -> None:
        """Queue one reduction after source producers and metadata are ready on the caller stream."""
        if self.submitted:
            raise ValueError("ordered reduction has already been submitted")
        self.submitted = True
        torch.npu.current_stream(self.metadata.device).wait_event(self.initialized)
        torch.ops.hyper_parallel.routed_experts_ordered_sum(self.sources, self.outputs, self.metadata)


@torch.no_grad()
def prepare_ordered_sum(groups: Sequence[Sequence[torch.Tensor]], outputs: Sequence[torch.Tensor],
                        local: Sequence[bool]) -> OrderedSumWork:
    """Prepare a single-launch ordered sum without reading device values.

    Args:
        groups: Source tensors per output, with local contribution first when present,
            followed by remote contributions in their required summation order.
        outputs: Independent contiguous BF16 destinations; sources must be FP32.
        local: Whether the first source initializes the accumulator by copy. Otherwise
            it starts at positive zero, including when a group has no contributions.

    Returns:
        A one-use frame. Its caller must order all source producers before run().
        Metadata is uploaded during preparation, before communication admission.
    """
    _load_native()
    if not outputs or len(groups) != len(outputs) or len(local) != len(outputs):
        raise ValueError("ordered reductions require matching source groups, outputs and local flags")
    sources, offsets = [], [0]
    for group in groups:
        sources.extend(group)
        offsets.append(len(sources))
    metadata = torch.ops.hyper_parallel.routed_experts_ordered_sum_prepare(
        sources, outputs, offsets, tuple(int(value) for value in local))
    initialized = torch.npu.Event()
    initialized.record(torch.npu.current_stream(metadata.device))
    return OrderedSumWork(tuple(sources), tuple(outputs), metadata, initialized)
