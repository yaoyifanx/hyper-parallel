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
"""Clipped FP32 SwiGLU arithmetic over ordinary Torch and vendor CANN operators."""

import math
from typing import Any

# Torch is the native framework; the platform-agnostic lint rule predates this component.
# pylint: disable=forbidden-backend-import
import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional


def _validate_projection(preact: torch.Tensor, limit: float) -> None:
    """Validate metadata without reading device values or synchronizing a stream."""
    if preact.ndim < 1 or preact.shape[-1] == 0 or preact.shape[-1] % 2:
        raise ValueError("clipped SwiGLU requires a positive, even packed projection width")
    if not preact.is_floating_point():
        raise ValueError("clipped SwiGLU requires floating projections")
    if isinstance(limit, bool) or not isinstance(limit, (float, int)) or not math.isfinite(limit):
        raise ValueError("clipped SwiGLU limit must be finite")


def _eager_values(preact: torch.Tensor, limit: float) -> torch.Tensor:
    """Preserve FP32 projection arithmetic on backends without the vendor op."""
    gate, up = preact.float().chunk(2, dim=-1)
    if limit > 0:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return (functional.silu(gate) * up).to(preact.dtype)


def _gradient(gradient: torch.Tensor, preact: torch.Tensor, limit: float,
              *, multiply_mask: bool) -> torch.Tensor:
    """Keep SiLU backward's multiplication order and the caller's clamp zero rule."""
    raw_gate, raw_up = preact.float().chunk(2, dim=-1)
    if preact.device.type == "npu":
        raw_gate, raw_up = raw_gate.contiguous(), raw_up.contiguous()
    mid = gradient.float()
    gate, up = raw_gate, raw_up
    if limit > 0:
        gate, up = raw_gate.clamp(max=limit), raw_up.clamp(-limit, limit)
    grad_gate = torch.ops.aten.silu_backward(mid * up, gate)
    grad_up = mid * functional.silu(gate)
    if limit > 0:
        gate_mask = raw_gate <= limit
        up_mask = (raw_up >= -limit) & (raw_up <= limit)
        if multiply_mask:
            grad_gate.mul_(gate_mask)
            grad_up.mul_(up_mask)
        else:
            # Clamp's autograd uses where: a rejected value becomes positive zero.
            grad_gate = torch.where(gate_mask, grad_gate, 0.0)
            grad_up = torch.where(up_mask, grad_up, 0.0)
    if preact.device.type != "npu":
        return torch.cat((grad_gate, grad_up), dim=-1).to(preact.dtype)
    # Convert directly into the two destination views instead of materializing a FP32 concatenation.
    result = torch.empty_like(preact, memory_format=torch.contiguous_format)
    result_gate, result_up = result.chunk(2, dim=-1)
    result_gate.copy_(grad_gate)
    result_up.copy_(grad_up)
    return result


def _cann_values(preact: torch.Tensor, limit: float) -> torch.Tensor:
    """Call the vendor op without loading any HyperParallel native payload."""
    # TorchNPU is optional for CPU/GPU users of this tensor primitive.
    import torch_npu  # pylint: disable=import-outside-toplevel
    # Adding negative zero preserves both signs of an unclipped zero up value.
    return torch_npu.npu_clipped_swiglu(
        preact, dim=-1, alpha=1.0, limit=limit, bias=-0.0, interleaved=False,
    )


class _ClippedSwiGLU(torch.autograd.Function):
    """Use CANN's packed forward while retaining the original FP32 derivative."""

    @staticmethod
    def forward(ctx: Any, preact: torch.Tensor, limit: float) -> torch.Tensor:
        """Save only the packed projections, with no expert or communication state."""
        ctx.save_for_backward(preact)
        ctx.limit = limit
        return _cann_values(preact, limit)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Match the standard clamp autograd masks rather than introducing a new derivative."""
        (preact,) = ctx.saved_tensors
        return _gradient(gradient, preact, ctx.limit, multiply_mask=False), None


def clipped_swiglu(preact: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply clipped SwiGLU to packed gate/up projections with FP32 internal math.

    Args:
        preact: Floating tensor ending in gate channels followed by up channels.
        limit: Positive gate upper bound and symmetric up bound; nonpositive disables clipping.

    Returns:
        Activated values with half the projection width and the input dtype.

    Note:
        The NPU path uses CANN ClippedSwiglu, with alpha=1, bias=-0 and noninterleaved inputs.
        Its first-order backward retains the SiLU derivative's original arithmetic order;
        higher-order differentiation through the vendor path is not supported.
    """
    _validate_projection(preact, limit)
    vendor_dtype = preact.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if preact.device.type == "npu" and vendor_dtype and preact.numel() and limit > 0:
        if torch.is_grad_enabled() and preact.requires_grad:
            return _ClippedSwiGLU.apply(preact, float(limit))
        return _cann_values(preact, float(limit))
    return _eager_values(preact, limit)


@torch.no_grad()
def clipped_swiglu_backward(gradient: torch.Tensor, preact: torch.Tensor, limit: float) -> torch.Tensor:
    """Differentiate packed projections using the explicit expert backward's clamp masks.

    Args:
        gradient: Intermediate gradient, with half the last dimension of ``preact``.
        preact: Saved floating packed gate/up projections.
        limit: Positive clipping bound; nonpositive disables clipping.

    Returns:
        Projection gradients with the input shape and dtype. Multiplicative masks
        preserve the signed zeros of the existing explicit MoonEP backward.
    """
    _validate_projection(preact, limit)
    expected = (*preact.shape[:-1], preact.shape[-1] // 2)
    if gradient.shape != expected or gradient.device != preact.device or not gradient.is_floating_point():
        raise ValueError("clipped SwiGLU gradient must match the activated shape and device")
    return _gradient(gradient, preact, limit, multiply_mask=True)
