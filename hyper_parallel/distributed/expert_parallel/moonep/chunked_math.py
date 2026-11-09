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
"""Full expert gradient inputs and source reduction order for the Python pipeline."""

from __future__ import annotations

from itertools import accumulate

# Torch is the native framework for this component.
# pylint: disable=forbidden-backend-import
import torch

from hyper_parallel.distributed.expert_parallel.moonep.chunked_plan import ChunkLayout


class ExpertArena:
    """Store each chunk once in full expert-major tensors, without retaining weights."""

    def __init__(self, chunks: ChunkLayout, counts: tuple[int, ...]) -> None:
        """Compile CPU-only chunk spans from the already agreed full expert counts."""
        bases = list(accumulate(counts, initial=0))
        used = [0] * len(counts)
        self.rows, self.ranges, self.spans = sum(counts), [], []
        for part in chunks:
            previous, ranges = 0, []
            for local, end in enumerate(part.spec.boundaries):
                expert = part.spec.weight_begin + local
                size = end - previous
                if size:
                    ranges.append((previous, bases[expert] + used[expert], size))
                    used[expert] += size
                previous = end
            self.ranges.append(ranges)
            first = ranges[0][1] if ranges else 0
            contiguous = all(target == first + source for source, target, _ in ranges)
            self.spans.append((first, previous) if contiguous else None)
        if tuple(used) != tuple(counts):
            raise ValueError('expert arena must cover every full expert row exactly once')

    def view(self, value: torch.Tensor, ordinal: int) -> torch.Tensor | None:
        """Return a contiguous chunk view when all of its expert segments are adjacent."""
        span = self.spans[ordinal]
        return None if span is None else value[span[0]:span[0] + span[1]]

    def store(self, value: torch.Tensor, ordinal: int, source: torch.Tensor) -> torch.Tensor:
        """Store one chunk and return a contiguous tensor usable by the next GEMM."""
        view = self.view(value, ordinal)
        if view is not None:
            if view.data_ptr() != source.data_ptr() or view.stride() != source.stride():
                view.copy_(source)
            return view
        for offset, target, size in self.ranges[ordinal]:
            value[target:target + size].copy_(source[offset:offset + size])
        return source

    def receive_views(self, value: torch.Tensor, chunks: ChunkLayout) -> list[torch.Tensor | None]:
        """Receive directly only when both peer/expert layout and destination span are contiguous."""
        return [self.view(value, ordinal) if part.receive_order is None else None
                for ordinal, part in enumerate(chunks)]


def source_gradients(gradient: torch.Tensor, values: torch.Tensor, probabilities: torch.Tensor,
                     indices: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute weighted expert gradients and FP32 route-score derivatives for one entry block.

    Args:
        gradient: Output gradients gathered once per source choice, shaped ``[entries, hidden]``.
        values: Saved expert outputs in the same entry order and dtype as ``gradient``.
        probabilities: FP32 router weights, shaped ``[entries, 1]``.
        indices: Cached INT32 identity permutation for this block on NPU; unused on CPU.

    Returns:
        Weighted gradients in the input dtype and FP32 score derivatives shaped ``[entries, 1]``.

    Note:
        Treating every entry as an independent TopK=1 token lets the CANN reverse
        combine operate on a partial route table without constructing FP32 hidden tables.
    """
    if gradient.device.type == 'npu':
        return torch.ops.npu.npu_moe_token_unpermute_grad(values, gradient, indices, probabilities)
    selected = gradient.float()
    return ((selected * probabilities).to(gradient.dtype),
            (selected * values.float()).sum(-1, keepdim=True))
