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

# Torch is the native framework for this component.
# pylint: disable=forbidden-backend-import
import torch

from hyper_parallel.distributed.expert_parallel.moonep.chunked_plan import DeviceChunk


def view_rows(value: torch.Tensor, chunk: DeviceChunk) -> torch.Tensor:
    """Return the chunk's contiguous rows in the complete expert activation cache."""
    begin, rows = chunk.spec.arena_row_span
    return value[begin:begin + rows]


def store_rows(value: torch.Tensor, chunk: DeviceChunk, source: torch.Tensor) -> torch.Tensor:
    """Write cached rows only when their storage and layout differ from the source."""
    view = view_rows(value, chunk)
    if view.data_ptr() != source.data_ptr() or view.stride() != source.stride():
        view.copy_(source)
    return view


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
