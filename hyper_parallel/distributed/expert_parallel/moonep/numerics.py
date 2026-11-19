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
"""不保存专家权重的两次前向/四次反向GMM与激活导数。"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
from torch.nn import functional

from hyper_parallel.components.functional.clipped_activation import clipped_swiglu, clipped_swiglu_backward


@dataclass(frozen=True)
class GroupedWeightViews:
    """Uniform expert matrix views without materializing their stacked home weights."""

    matrices: tuple[torch.Tensor, ...] = ()
    parts: tuple[torch.Tensor, ...] = ()
    rows_per_part: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.parts:
            if self.matrices or len(self.parts) != len(self.rows_per_part):
                raise ValueError("grouped weight parts require one row count per batch and no matrix list")
            first = self.parts[0]
            if first.ndim != 3 or any(value.ndim != 3 or value.shape[1:] != first.shape[1:]
                                      or value.device != first.device or value.dtype != first.dtype
                                      for value in self.parts):
                raise ValueError("grouped weight parts must have uniform matrix shapes, dtype, and device")
            if (not 1 <= sum(value.shape[0] for value in self.parts) <= 128
                    or any(isinstance(rows, bool) or not isinstance(rows, int) or rows < 0
                           or (rows > 0 and not value.shape[0])
                           for value, rows in zip(self.parts, self.rows_per_part))):
                raise ValueError("grouped weight parts require admitted expert and nonnegative row counts")
            return
        if not 1 <= len(self.matrices) <= 128:
            raise ValueError("grouped weight views require between one and 128 expert matrices")
        first = self.matrices[0]
        if first.ndim != 2 or any(value.ndim != 2 or value.shape != first.shape
                                  or value.device != first.device or value.dtype != first.dtype
                                  for value in self.matrices):
            raise ValueError("grouped weight views must have one shape, dtype, and device")

    @property
    def shape(self) -> torch.Size:
        """Return the logical stacked shape without allocating that stack."""
        if self.parts:
            return torch.Size((sum(value.shape[0] for value in self.parts), *self.parts[0].shape[1:]))
        return torch.Size((len(self.matrices), *self.matrices[0].shape))

    def transpose(self, dim0: int, dim1: int) -> "GroupedWeightViews":
        """Transpose expert matrices while preserving the grouped expert axis."""
        if {dim0 % 3, dim1 % 3} != {1, 2}:
            raise ValueError("grouped weight views only transpose the last two dimensions")
        if self.parts:
            return GroupedWeightViews(parts=tuple(value.transpose(1, 2) for value in self.parts),
                                      rows_per_part=self.rows_per_part)
        return GroupedWeightViews(tuple(value.t() for value in self.matrices))

    def record_stream(self, stream: Any) -> None:
        """Protect each distinct home/remote storage once on its consumer stream."""
        seen = set()
        for value in self.parts or self.matrices:
            pointer = value.untyped_storage().data_ptr()
            if pointer not in seen:
                value.record_stream(stream)
                seen.add(pointer)


GroupedWeights = Union[torch.Tensor, GroupedWeightViews]
GroupedOp = Callable[[torch.Tensor, GroupedWeights, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ExpertSaved:
    """保存反向必要激活；字段中没有专家权重。

    Example:
        rank 1 的 [d,b,c,d] 专家主序输入产生 x[4,2]、preact[4,2]、
        intermediate[4,1] 和 boundaries=[1,2,4]，权重随后可释放。
    """

    x: torch.Tensor
    preact: torch.Tensor
    intermediate: torch.Tensor
    boundaries: torch.Tensor
    limit: float
    activation: str = "clipped"


@torch.no_grad()
def expert_forward_values(
    x: torch.Tensor,
    gate_up: GroupedWeights,
    down: GroupedWeights,
    boundaries: torch.Tensor,
    limit: float,
    gmm: GroupedOp,
    activation: str = "clipped",
) -> tuple[torch.Tensor, ExpertSaved]:
    """执行两次 GMM，并只返回保存激活的反向上下文。

    Args:
        x: [C,H] 专家主序输入。
        gate_up: [A,H,2I] 当前物化的权重。
        down: [A,I,H] 当前物化的权重，与 gate_up 专家顺序相同。
        boundaries: [A] int64 累计分组边界，已在设备上。
        limit: SwiGLU 限幅；非正数表示不限幅。
        gmm: 不保存 autograd 图的 group_type=0 grouped GEMM 接口。
        activation: 普通 swiglu 或 clipped；两条分支保留各自原有的舍入边界。

    Returns:
        [C,H] 专家输出，以及仅含激活/边界/限幅配置的 ExpertSaved。

    Example:
        x=[d,b,c,d]、boundaries=[1,2,4] 时，依次计算 e0/e2/e3；
        输出为 [F0(d),F2(b),F3(c),F3(d)]，等待外部恢复接收顺序。
    """
    preact = gmm(x, gate_up, boundaries)
    if activation == "swiglu":
        intermediate = swiglu_values(preact)
    elif activation == "clipped":
        intermediate = clipped_swiglu(preact, limit).to(x.dtype)
    else:
        raise ValueError("未知的专家激活类型")
    output = gmm(intermediate, down, boundaries)
    return output, ExpertSaved(x, preact, intermediate, boundaries, limit, activation)


@torch.no_grad()
def expert_backward_values(
    grad_output: torch.Tensor,
    saved: ExpertSaved,
    gate_up: GroupedWeights,
    down: GroupedWeights,
    gmm: GroupedOp,
    gmm_weight_grad: GroupedOp,
    on_weight_gradients: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """用重新物化的权重计算四次 GMM 反向，不重跑前向矩阵乘。

    Args:
        grad_output: [C,H] 专家主序输出梯度。
        saved: 对应同一层、同一 microbatch 前向保存的激活。
        gate_up: 与前向数值/版本相同的 [A,H,2I] 重建权重。
        down: 与前向数值/版本相同的 [A,I,H] 重建权重。
        gmm: 输入梯度所需的 group_type=0 接口。
        gmm_weight_grad: 输入 X[N,J],dY[N,K]，返回 [A,J,K] 分组梯度。
        on_weight_gradients: 两份 dW 就绪后调用；例如先发 EP 回传再计算 dX GMM。

    Returns:
        dx[C,H]、dgate[A,H,2I]、ddown[A,I,H]；后两者尚未归还 home。

    Example:
        rank 1 产生 e0/e2/e3 三组梯度；外部先将 e0 梯度回传 rank 0，
        e2/e3 留作本地参数贡献，再交给 FSDP 归约。
    """
    grad_output = grad_output.contiguous()
    boundaries = saved.boundaries
    grad_mid = gmm(grad_output, down.transpose(-1, -2), boundaries)
    grad_down = gmm_weight_grad(saved.intermediate, grad_output, boundaries)
    grad_preact = activation_gradient(grad_mid, saved.preact, saved.limit, saved.activation)
    if on_weight_gradients is None:
        grad_x = gmm(grad_preact, gate_up.transpose(-1, -2), boundaries)
        grad_gate_up = gmm_weight_grad(saved.x, grad_preact, boundaries)
    else:
        grad_gate_up = gmm_weight_grad(saved.x, grad_preact, boundaries)
        on_weight_gradients(grad_gate_up, grad_down)
        grad_x = gmm(grad_preact, gate_up.transpose(-1, -2), boundaries)
    return grad_x, grad_gate_up, grad_down


@torch.no_grad()
def activation_gradient(gradient: torch.Tensor, preact: torch.Tensor, limit: float,
                        activation: str = "clipped") -> torch.Tensor:
    """Differentiate one expert row identically for full and chunked execution.

    Args:
        gradient: Intermediate activation gradient [rows, intermediate].
        preact: Saved gate/up projection [rows, 2 * intermediate].
        limit: Positive clipping bound, or a nonpositive value to disable clipping.
        activation: The model's ``clipped`` or ordinary ``swiglu`` activation.

    Returns:
        Gate/up projection gradients with the shape and dtype of ``preact``.

    Note:
        Clipped and chunked experts share CANN activation math and the same FP32
        derivative order, inclusive masks, and final projection-gradient rounding.
    """
    if activation == "swiglu":
        return swiglu_grad(gradient, preact)
    if activation != "clipped":
        raise ValueError("未知的专家激活类型")
    return clipped_swiglu_backward(gradient, preact, limit)


def swiglu_values(preact: torch.Tensor) -> torch.Tensor:
    """复用基线相同的SwiGLU；例如[C,1536]变为[C,768]且保持BF16舍入。"""
    if preact.device.type == "npu":
        # 可选昇腾依赖只在实际设备路径加载，CPU逻辑测试不要求安装。
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_swiglu(preact, dim=-1)
    gate, up = preact.chunk(2, dim=-1)
    return functional.silu(gate) * up


def swiglu_grad(gradient: torch.Tensor, preact: torch.Tensor) -> torch.Tensor:
    """复用现有SwiGLU反向；例如[C,I]梯度恢复为两投影的[C,2I]梯度。"""
    if preact.device.type == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_swiglu_backward(gradient.contiguous(), preact, dim=-1)
    gate, up = preact.chunk(2, dim=-1)
    return torch.cat((torch.ops.aten.silu_backward(gradient * up, gate),
                      gradient * functional.silu(gate)), dim=-1)


@torch.no_grad()
def grouped_values(x: torch.Tensor, weight: GroupedWeights,
                   boundaries: torch.Tensor) -> torch.Tensor:
    """不构造权重autograd缓存的分组乘法；例如两组输入分别乘weight[0]/weight[1]。"""
    if x.shape[0] == 0:
        return x.new_empty((0, weight.shape[-1]))
    if x.device.type == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        if isinstance(weight, GroupedWeightViews) and weight.parts:
            return _grouped_parts(x, weight, boundaries, torch_npu.npu_grouped_matmul)
        return torch_npu.npu_grouped_matmul(
            x=[x], weight=list(weight.matrices) if isinstance(weight, GroupedWeightViews) else [weight],
            bias=[], group_list=boundaries,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
    begin = 0
    outputs = []
    matrices = weight
    if isinstance(weight, GroupedWeightViews):
        matrices = (tuple(matrix for part in weight.parts for matrix in part.unbind(0))
                    if weight.parts else weight.matrices)
    for index, end in enumerate(boundaries.tolist()):
        outputs.append(x[begin:end] @ matrices[index])
        begin = end
    return torch.cat(outputs)


def _grouped_parts(x: torch.Tensor, weight: GroupedWeightViews, boundaries: torch.Tensor,
                   operator: Callable) -> torch.Tensor:
    """Use batched home/remote GEMMs without a per-expert Python weight list."""
    if sum(weight.rows_per_part) != x.shape[0] or boundaries.numel() != weight.shape[0]:
        raise ValueError("grouped weight partitions do not cover the input and expert boundaries")
    nonempty = sum(rows > 0 for rows in weight.rows_per_part)
    result = None if nonempty == 1 else x.new_empty((x.shape[0], weight.shape[-1]))
    row_begin = group_begin = 0
    for part, rows in zip(weight.parts, weight.rows_per_part):
        group_end = group_begin + part.shape[0]
        if rows:
            groups = boundaries[group_begin:group_end]
            if row_begin:
                groups = groups - row_begin
            value = operator(x=[x[row_begin:row_begin + rows]], weight=[part], bias=[],
                             group_list=groups, split_item=3, group_type=0, group_list_type=0)[0]
            if result is None:
                return value
            result[row_begin:row_begin + rows].copy_(value)
        row_begin += rows
        group_begin = group_end
    return result


@torch.no_grad()
def grouped_weight_grad(x: torch.Tensor, gradient: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """group_type=2聚合每专家dW；例如X[C,H]与dU[C,2I]生成[A,H,2I]。"""
    if x.shape[0] == 0:
        return x.new_zeros((boundaries.numel(), x.shape[1], gradient.shape[1]))
    if x.device.type == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        return torch_npu.npu_grouped_matmul(
            x=[x.transpose(-1, -2)], weight=[gradient.contiguous()], bias=[], group_list=boundaries,
            split_item=2, group_type=2, group_list_type=0,
        )[0]
    begin = 0
    outputs = []
    for end in boundaries.tolist():
        outputs.append(x[begin:end].T @ gradient[begin:end])
        begin = end
    return torch.stack(outputs)


@torch.no_grad()
def grouped_weight_grad_add(target: torch.Tensor, x: torch.Tensor, derivative: torch.Tensor,
                            boundaries: torch.Tensor, *, transposed: bool = False) -> None:
    """Accumulate home dW directly in the parameter's contiguous FP32 storage.

    BF16/FP16 operands retain the forward rounding. No BF16 dW is materialized.
    Empty groups preserve previous microsteps, including an entirely empty batch.
    """
    if target.dtype != torch.float32 or not target.is_contiguous():
        raise ValueError("grouped wgrad accumulation requires contiguous FP32 parameter storage")
    if x.dtype != derivative.dtype or x.shape[0] != derivative.shape[0]:
        raise ValueError("grouped wgrad operands must have matching dtype and rows")
    if target.shape[0] != boundaries.numel():
        raise ValueError("one grouped boundary is required per home expert")
    left, right = (derivative, x) if transposed else (x, derivative)
    if target.shape[1:] != (left.shape[1], right.shape[1]):
        raise ValueError("grouped wgrad destination does not match the parameter layout")
    if not x.shape[0]:
        return
    if x.device.type == "npu":
        # Optional Ascend dependency; CPU numerical tests use the same FP32 contract.
        import torch_npu  # pylint: disable=import-outside-toplevel
        torch_npu.npu_grouped_matmul_add_(target, left, right, boundaries)
        return
    start = 0
    for index, end in enumerate(boundaries.tolist()):
        target[index].addmm_(left[start:end].float().T, right[start:end].float())
        start = end
