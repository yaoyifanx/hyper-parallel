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
"""Independent cube tasks over actual expert ranges and bounded vector reductions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
import math
from typing import Any, Optional

import numpy as np
import torch

from hyper_parallel.core.multicore.modules.routed_experts.forward import ForwardPipeline, PipelineForwardWork
from hyper_parallel.core.multicore.modules.routed_experts.frame import (
    BackwardWork, ExpertState, ForwardWork, GradientWork, LinearWork,
)
from hyper_parallel.core.multicore.modules.routed_experts.readiness import (
    ExpertReadiness, ProducerFence, producer_progress, upload_metadata, upload_metadata_batch,
)
from hyper_parallel.core.multicore.modules.routed_experts.transport import _load_native


TensorAllocator = Callable[[str, tuple[int, ...], torch.dtype], Optional[torch.Tensor]]


@lru_cache(maxsize=128)
def _linear_config(rows: int, cols: int, inner: int, transposed: bool) -> torch.Tensor:
    """Cache immutable host tiling, without retaining any model or device storage."""
    _load_native()
    config, workspace = torch.ops.hyper_parallel.routed_experts_gemm_config(rows, cols, inner, transposed)
    if workspace:
        raise RuntimeError("the independent cube-only kernel must not allocate KFC workspace")
    config_bytes = torch.ops.hyper_parallel.routed_experts_gemm_config_bytes()
    _check_tensor(config, (config_bytes,), torch.uint8, torch.device("cpu"), "linear configuration")
    return config


@lru_cache(maxsize=128)
def _gradient_config(features: int, cols: int, max_tokens: int) -> torch.Tensor:
    """Cache the FP32 gradient tiling shared by equivalent actual group extents."""
    _load_native()
    config, workspace = torch.ops.hyper_parallel.routed_experts_gradient_config(features, cols, max_tokens, 128, 256)
    if workspace:
        raise RuntimeError("the independent gradient kernel must not allocate KFC workspace")
    return config


def _validate_task_shape(boundaries: tuple[int, ...], *widths: int) -> None:
    """Reject malformed host task dimensions before constructing any device metadata."""
    if any(isinstance(end, bool) or not isinstance(end, int) or end <= begin
           for begin, end in zip((0, *boundaries[:-1]), boundaries)):
        raise ValueError("compute boundaries must describe strictly positive expert ranges")
    if boundaries and boundaries[-1] > 2**20:
        raise ValueError("compute supports at most 2**20 packed token selections")
    if any(isinstance(width, bool) or not isinstance(width, int) or width <= 0 or width > 32768
           for width in widths):
        raise ValueError("compute widths must be positive integers at most 32768")


def linear_tasks(boundaries: tuple[int, ...], cols: int, tile_rows: int = 128,
                 tile_cols: int = 256) -> torch.Tensor:
    """Build only nonempty matrix tiles; no slot-capacity dimension enters the grid."""
    _validate_task_shape(boundaries, cols, tile_rows, tile_cols)
    if not boundaries:
        return torch.from_numpy(np.empty((0, 5), dtype=np.int64))
    ends = np.asarray(boundaries, dtype=np.int64)
    lengths = np.diff(ends, prepend=0)
    counts = (lengths + tile_rows - 1) // tile_rows
    experts = np.repeat(np.arange(len(boundaries), dtype=np.int64), counts)
    ordinal = np.arange(len(experts), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    rows = (ends - lengths)[experts] + ordinal * tile_rows
    columns = np.arange(0, cols, tile_cols, dtype=np.int64)
    table = np.empty((len(rows), len(columns), 5), dtype=np.int64)
    table[:, :, 0] = experts[:, None]
    table[:, :, 1] = rows[:, None]
    table[:, :, 2] = np.minimum(tile_rows, ends[experts] - rows)[:, None]
    table[:, :, 3] = columns
    table[:, :, 4] = np.minimum(tile_cols, cols - columns)
    return torch.from_numpy(table.reshape(-1, 5))


@lru_cache(maxsize=8)
def _gradient_tile_template(features: int, cols: int) -> np.ndarray:
    """Keep at most eight read-only geometric templates, each bounded by one MiB.

    Callers validate widths before lookup. Dynamic expert/token fields and returned
    tensor storage are never cached, so modifying one result cannot affect another.
    """
    rows = np.arange(0, features, 128, dtype=np.int64)
    columns = np.arange(0, cols, 256, dtype=np.int64)
    table = np.empty((len(rows), len(columns), 4), dtype=np.int64)
    table[:, :, 0] = rows[:, None]
    table[:, :, 1] = np.minimum(128, features - rows)[:, None]
    table[:, :, 2] = columns
    table[:, :, 3] = np.minimum(256, cols - columns)
    template = table.reshape(-1, 4)
    template.setflags(write=False)
    return template


def gradient_tasks(boundaries: tuple[int, ...], features: int, cols: int) -> torch.Tensor:
    """Build a unique FP32 output tile per expert, reducing its complete token span."""
    _validate_task_shape(boundaries, features, cols)
    if not boundaries:
        return torch.from_numpy(np.empty((0, 7), dtype=np.int64))
    template = _gradient_tile_template(features, cols)
    ends = np.asarray(boundaries, dtype=np.int64)
    lengths = np.diff(ends, prepend=0)
    table = np.empty((len(boundaries), len(template), 7), dtype=np.int64)
    table[:, :, 0] = np.arange(len(boundaries), dtype=np.int64)[:, None]
    table[:, :, 1:5] = template
    table[:, :, 5] = (ends - lengths)[:, None]
    table[:, :, 6] = lengths[:, None]
    return torch.from_numpy(table.reshape(-1, 7))


def _check_tensor(value: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype,
                  device: torch.device, name: str, *, contiguous: bool = True) -> None:
    """Validate host-visible storage before a phase's collective admission."""
    if value.shape != shape or value.dtype != dtype or value.device != device:
        raise ValueError(f"{name} must have shape {shape}, dtype {dtype}, and device {device}")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if value.numel() and (value.data_ptr() == 0 or value.untyped_storage().nbytes()
                          < (value.storage_offset() + value.numel()) * value.element_size()):
        raise ValueError(f"{name} storage has been invalidated or is undersized")


def _check_output_aliases(output: torch.Tensor, *inputs: torch.Tensor,
                          allow_disjoint_storage: bool = False) -> None:
    """Reject storage aliases except explicitly admitted disjoint dense ranges."""
    if not output.numel():
        return
    start = output.data_ptr()
    end = start + output.numel() * output.element_size()
    for value in inputs:
        if not value.numel() or value.untyped_storage().data_ptr() != output.untyped_storage().data_ptr():
            continue
        if allow_disjoint_storage and output.is_contiguous() and value.is_contiguous():
            other = value.data_ptr()
            if end <= other or other + value.numel() * value.element_size() <= start:
                continue
        raise ValueError("native compute output must not alias an input storage")


class NativeExpertCompute:
    """Own one execution's small tile descriptors, without expert weights or payload pools."""

    def __init__(self, boundaries: tuple[int, ...], device: torch.device, *,
                 vector_stream: Any = None, vector_workers: int = 8,
                 swiglu_limit: float | None = None) -> None:
        """Freeze strictly nonempty active ranges that cover the packed expert input."""
        if any(isinstance(end, bool) or not isinstance(end, int) or end <= begin
               for begin, end in zip((0, *boundaries[:-1]), boundaries)):
            raise ValueError("native compute boundaries must describe strictly positive expert ranges")
        if boundaries and boundaries[-1] > 2**20:
            raise ValueError("native compute supports at most 2**20 packed token selections")
        if device.type != "npu":
            raise ValueError("native routed experts require an NPU device")
        if swiglu_limit is not None and (isinstance(swiglu_limit, bool)
                                         or not isinstance(swiglu_limit, (int, float))
                                         or not math.isfinite(swiglu_limit) or swiglu_limit <= 0):
            raise ValueError("swiglu_limit must be a finite positive number or None")
        self.boundaries, self.device = boundaries, device
        self.swiglu_limit = None if swiglu_limit is None else float(swiglu_limit)
        self.prefix = upload_metadata(torch.tensor((0, *boundaries), dtype=torch.int64), device)
        self.empty_workspace = torch.empty(0, dtype=torch.uint8, device=device)
        self.initialized = torch.npu.Event()
        self.initialized.record(torch.npu.current_stream(device))
        self.linear_metadata: dict[tuple, tuple] = {}
        self.gradient_metadata: dict[tuple, tuple] = {}
        self.forward_pipeline = ForwardPipeline(
            boundaries, device, vector_stream, vector_workers, self.swiglu_limit)

    def _activate(self, preact: torch.Tensor, output: torch.Tensor) -> None:
        """Fill the admitted activation buffer using the model's SwiGLU rule."""
        limit = self.swiglu_limit
        if limit is None:
            torch.ops.hyper_parallel.routed_experts_swiglu(preact, output)
            return
        torch.ops.hyper_parallel.routed_experts_clipped_swiglu(preact, output, limit)

    def _activation_gradient(self, derivative: torch.Tensor, preact: torch.Tensor,
                             output: torch.Tensor, *, consume_preactivation: bool = False) -> None:
        """Differentiate the FP32 clipped activation before BF16 projection gradients."""
        limit = self.swiglu_limit
        if consume_preactivation and limit is not None:
            raise ValueError("preactivation consumption requires unclipped SwiGLU")
        if limit is None:
            if consume_preactivation:
                if output.data_ptr() != preact.data_ptr():
                    raise ValueError("consumed preactivation must be the exact derivative output")
                torch.ops.hyper_parallel.routed_experts_swiglu_backward_consume(derivative, preact)
            else:
                torch.ops.hyper_parallel.routed_experts_swiglu_backward(derivative, preact, output)
            return
        torch.ops.hyper_parallel.routed_experts_clipped_swiglu_backward(derivative, preact, output, limit)

    def _check_input(self, value: torch.Tensor, name: str) -> None:
        """Check one packed matrix without reading device contents."""
        if value.ndim != 2:
            raise ValueError(f"{name} must be a packed matrix")
        _validate_task_shape(self.boundaries, value.shape[1])
        if value.shape[1] % 16:
            raise ValueError(f"{name} width must be divisible by 16")
        rows = self.boundaries[-1] if self.boundaries else 0
        _check_tensor(value, (rows, value.shape[1]), torch.bfloat16, self.device, name)
        if value.numel() and value.data_ptr() % 32:
            raise ValueError(f"{name} address must be aligned to 32 bytes")

    def _allocate(self, allocator: TensorAllocator | None, name: str, shape: tuple[int, ...],
                  dtype: torch.dtype) -> torch.Tensor:
        """Use an admitted arena slot when supplied, preserving exact tensor contracts."""
        value = None if allocator is None else allocator(name, shape, dtype)
        if value is None:
            value = torch.empty(shape, dtype=dtype, device=self.device)
        _check_tensor(value, shape, dtype, self.device, name)
        return value

    def _check_linear_weights(self, x: torch.Tensor, weights: tuple[torch.Tensor, ...], cols: int) -> bool:
        """Validate current expert matrices in their original short-circuit order."""
        if weights and weights[0].ndim != 2:
            raise ValueError("linear weights must be matrices")
        transposed = bool(weights and weights[0].stride(-2) == 1)
        for weight in weights:
            _check_tensor(weight, (x.shape[1], cols), x.dtype, self.device, "linear weight", contiguous=False)
            contiguous = weight.t().is_contiguous() if transposed else weight.is_contiguous()
            if not contiguous:
                raise ValueError("linear weights must share one contiguous or transposed layout")
        return transposed

    def _prepare_linear_metadata(self, x: torch.Tensor, cols: int, transposed: bool,
                                 publish: bool, current: Any) -> tuple:
        """Prepare or join cached task metadata, retaining late-publish ordering."""
        tile_rows, tile_cols = 128, min(512, cols)
        key = x.shape[1], cols, transposed
        metadata = self.linear_metadata.get(key)
        if metadata is None:
            host_tasks = linear_tasks(self.boundaries, cols, tile_rows, tile_cols)
            cores = min(24, host_tasks.shape[0])
            expected = np.zeros((cores, len(self.boundaries)), dtype=np.int32)
            expected[np.arange(host_tasks.shape[0]) % cores, host_tasks.numpy()[:, 0]] = 1
            expected = torch.from_numpy(expected)
            host_config = _linear_config(tile_rows, tile_cols, x.shape[1], transposed)
            if publish:
                tasks, config, expected_device = upload_metadata_batch((host_tasks, host_config, expected), self.device)
            else:
                tasks, config = upload_metadata_batch((host_tasks, host_config), self.device)
                expected_device = None
            initialized = torch.npu.Event()
            initialized.record(current)
            progress = (producer_progress(host_tasks, self.boundaries, cores, tile_rows,
                                          (cols + tile_cols - 1) // tile_cols) if publish else None)
            metadata = tasks, config, expected, expected_device, progress, initialized
            self.linear_metadata[key] = metadata
        tasks, config, expected, expected_device, progress, initialized = metadata
        current.wait_event(initialized)
        if publish and expected_device is None:
            expected_device = upload_metadata(expected, self.device)
            progress = producer_progress(linear_tasks(self.boundaries, cols, tile_rows, tile_cols),
                                         self.boundaries, expected.shape[0], tile_rows,
                                         (cols + tile_cols - 1) // tile_cols)
            initialized = torch.npu.Event()
            initialized.record(current)
            self.linear_metadata[key] = tasks, config, expected, expected_device, progress, initialized
        return tasks, config, expected, expected_device, progress, initialized

    def prepare_linear(self, x: torch.Tensor, weights: Sequence[torch.Tensor], cols: int, *,
                       publish: bool = False, output: torch.Tensor | None = None) -> LinearWork:
        """Allocate and validate a GEMM before admission.

        Args:
            x: Contiguous BF16 expert-major input.
            weights: Current leased matrices, one per active expert.
            cols: Output matrix width.
            publish: Allocate fresh per-expert output readiness cells.
            output: Optional contiguous BF16 destination with no input aliases.

        Returns:
            Work retaining all operands and metadata for exactly one submission.
        """
        self._check_input(x, "linear input")
        _validate_task_shape(self.boundaries, cols)
        if cols % 16 or len(weights) != len(self.boundaries):
            raise ValueError("linear columns must align to 16 and weights must cover the active experts")
        weights = tuple(weights)
        transposed = self._check_linear_weights(x, weights, cols)
        result = x.new_empty((x.shape[0], cols)) if output is None else output
        _check_tensor(result, (x.shape[0], cols), x.dtype, self.device, "linear output")
        if result.numel() and result.data_ptr() % 32:
            raise ValueError("linear output address must be aligned to 32 bytes")
        _check_output_aliases(result, x, *weights)
        current = torch.npu.current_stream(self.device)
        current.wait_event(self.initialized)
        if not weights:
            return LinearWork(self, x, weights, None, result, None, None,
                              self.empty_workspace, self.initialized, False, 0, None)
        tasks, config, expected, expected_device, progress, initialized = self._prepare_linear_metadata(
            x, cols, transposed, publish, current)
        # Each frame binds current FSDP storage; a prior execution's addresses are invalid.
        weight_table = torch.ops.hyper_parallel.routed_experts_weight_table(weights)
        _check_tensor(weight_table, (len(weights),), torch.int64, self.device, "linear weight table")
        cores = min(24, tasks.shape[0])
        schedule_stride = ((len(weights) * 4 * 8 + 63) // 64) * 64
        # Different prepared frames may run on different streams of this engine.
        schedule_workspace = torch.empty(cores * schedule_stride, dtype=torch.uint8, device=self.device)
        if schedule_workspace.data_ptr() % 64:
            raise ValueError("linear scheduling workspace must be aligned to 64 bytes")
        ready = (ExpertReadiness.create(expected, self.device, deferred_poll=True, expected_device=expected_device,
                                        progress=progress)
                 if publish else None)
        if ready is None:
            initialized = torch.npu.Event()
            initialized.record(current)
        else:
            initialized = ready.admitted
        return LinearWork(self, x, weights, weight_table, result, tasks, config, schedule_workspace,
                          initialized, transposed, cores, ready)

    def run_linear(self, work: LinearWork, dependency: ExpertReadiness | None = None) -> tuple:
        """Submit prepared GEMM storage; input readiness may arrive from another stream."""
        if dependency is not None:
            dependency.validate_polling()
        self._claim_unsubmitted_work(work)
        current = torch.npu.current_stream(self.device)
        current.wait_event(self.initialized)
        current.wait_event(work.initialized)
        if dependency is not None:
            current.wait_event(dependency.admitted)
            current.wait_event(dependency.poll_after.event)
        if work.ready is not None:
            current.wait_event(work.ready.admitted)
            # Persistent transport must not occupy vector workers while a
            # preceding vector operation is still producing this GEMM input.
            work.ready.poll_after.event.record(current)
        if work.weights:
            torch.ops.hyper_parallel.routed_experts_gemm(
                work.x, work.weights, work.tasks, work.result, work.config, work.schedule_workspace,
                work.transposed, work.cores, None if dependency is None else dependency.flags,
                None if dependency is None else (dependency.task_expected
                                                 if dependency.task_expected is not None else dependency.expected),
                None if work.ready is None else work.ready.flags, 60_000_000_000, work.weight_table)
        if work.ready is not None:
            work.ready.poll_after.queued = True
        return work.result, work.ready


    def prepare_gradient(self, x: torch.Tensor, dy: torch.Tensor, *,
                         output: torch.Tensor | None = None) -> GradientWork:
        """Prepare one FP32 partial with a unique writer per expert output tile.

        Args:
            x: Contiguous BF16 input with the admitted expert boundaries.
            dy: Contiguous BF16 derivative with the same packed token extent.
            output: Optional FP32 [active_experts, features, cols] destination.

        Returns:
            Work retaining independent gradient storage for one submission.
        """
        self._check_input(x, "gradient input")
        self._check_input(dy, "gradient derivative")
        shape = len(self.boundaries), x.shape[1], dy.shape[1]
        result = torch.empty(shape, dtype=torch.float32, device=self.device) if output is None else output
        _check_tensor(result, shape, torch.float32, self.device, "gradient output")
        if result.numel() and result.data_ptr() % 32:
            raise ValueError("gradient output address must be aligned to 32 bytes")
        _check_output_aliases(result, x, dy)
        current = torch.npu.current_stream(self.device)
        current.wait_event(self.initialized)
        if not self.boundaries:
            return GradientWork(self, x, dy, result, None, None, self.initialized, 0)
        key = x.shape[1], dy.shape[1]
        metadata = self.gradient_metadata.get(key)
        if metadata is None:
            maximum = max(end - begin for begin, end in zip((0, *self.boundaries[:-1]), self.boundaries))
            tasks, config = upload_metadata_batch(
                (gradient_tasks(self.boundaries, *key), _gradient_config(*key, maximum)), self.device)
            initialized = torch.npu.Event()
            initialized.record(current)
            metadata = tasks, config, initialized
            self.gradient_metadata[key] = metadata
        tasks, config, initialized = metadata
        current.wait_event(initialized)
        return GradientWork(self, x, dy, result, tasks, config, initialized, min(24, tasks.shape[0]))

    def prepare_mapped_gradient(self, x: torch.Tensor, dy: torch.Tensor,
                                outputs: Sequence[torch.Tensor], transposed: bool) -> GradientWork:
        """Prepare direct per-expert FP32 destinations without a compact home dW table."""
        self._check_input(x, "gradient input")
        self._check_input(dy, "gradient derivative")
        left, right = (dy, x) if transposed else (x, dy)
        if len(outputs) != len(self.boundaries):
            raise ValueError("mapped gradients require one destination per active expert")
        for output in outputs:
            _check_tensor(output, (left.shape[1], right.shape[1]), torch.float32, self.device, "mapped dW")
            _check_output_aliases(output, x, dy, allow_disjoint_storage=True)
        if not self.boundaries:
            return GradientWork(self, left, right, tuple(outputs), None, None, self.initialized, 0)
        tasks = upload_metadata(gradient_tasks(self.boundaries, left.shape[1], right.shape[1]), self.device)
        maximum = max(end - start for start, end in zip((0, *self.boundaries[:-1]), self.boundaries))
        config = upload_metadata(_gradient_config(left.shape[1], right.shape[1], maximum), self.device)
        pointers = upload_metadata(
            torch.tensor([value.data_ptr() for value in outputs], dtype=torch.int64), self.device)
        initialized = torch.npu.Event()
        initialized.record(torch.npu.current_stream(self.device))
        return GradientWork(self, left, right, tuple(outputs), tasks, config, initialized,
                            min(24, tasks.shape[0]), output_table=pointers, transposed=transposed)

    def run_gradient(self, work: GradientWork) -> torch.Tensor:
        """Submit an already allocated expert gradient and retain its input lifetimes."""
        self._claim_unsubmitted_work(work)
        current = torch.npu.current_stream(self.device)
        current.wait_event(self.initialized)
        current.wait_event(work.initialized)
        if work.cores:
            torch.ops.hyper_parallel.routed_experts_gradient(
                work.x, work.dy, self.prefix, work.tasks, work.result, work.config, work.cores)
        return work.result


    def _claim_unsubmitted_work(self, work: LinearWork | GradientWork | ForwardWork | BackwardWork) -> None:
        """Reject reuse because published readiness cells are valid for one phase only."""
        if work.owner is not self or work.submitted:
            raise ValueError("prepared compute work belongs to another execution or was already submitted")
        work.submitted = True

    @torch.no_grad()
    def prepare_forward(self, x: torch.Tensor, gate: Sequence[torch.Tensor], down: Sequence[torch.Tensor],
                        intermediate: int, *,
                        allocator: TensorAllocator | None = None) -> ForwardWork | PipelineForwardWork:
        """Allocate complete F/replay storage before its first payload is admitted.

        The returned output and readiness are available to prepare the inverse
        transport. The saved state owns X/Z after the work frame is released.
        The optional allocator provides activation/project scratch; saved Z is
        always allocated independently. Returning None requests a fresh tensor.
        """
        _load_native()
        _validate_task_shape(self.boundaries, intermediate)
        if intermediate % 16:
            raise ValueError("the expert intermediate width must be divisible by 16")
        if (self.forward_pipeline.enabled and x.ndim == 2 and self.forward_pipeline.supports(
                x.shape[1], intermediate, gate, down, len(self.boundaries))):
            return self.forward_pipeline.prepare(x, gate, down, intermediate, allocator)
        up = self.prepare_linear(x, gate, 2 * intermediate)
        activation = self._allocate(allocator, "activation", (x.shape[0], intermediate), x.dtype)
        output = self._allocate(allocator, "project", tuple(x.shape), x.dtype)
        _check_output_aliases(activation, x, up.result, output, *gate, *down)
        _check_output_aliases(output, x, up.result, *gate, *down)
        projection = self.prepare_linear(activation, down, x.shape[1], publish=True, output=output)
        state = ExpertState(x, up.result)
        return ForwardWork(self, activation, projection.result,
                           projection.ready, state, up, projection)

    @torch.no_grad()
    def run_forward(self, work: ForwardWork | PipelineForwardWork, dependency: ExpertReadiness | None = None) -> tuple:
        """Submit F/replay using caller-owned outputs and return only X/Z as saved state."""
        if isinstance(work, PipelineForwardWork):
            return self.forward_pipeline.run(work, dependency)
        self._claim_unsubmitted_work(work)
        self.run_linear(work.up, dependency)
        self._activate(work.state.preact, work.activation)
        self.run_linear(work.down)
        return work.output, work.state, work.ready


    @torch.no_grad()
    def prepare_backward(self, dv: torch.Tensor, state: ExpertState, gate: Sequence[torch.Tensor],
                         down: Sequence[torch.Tensor], *, allocator: TensorAllocator | None = None,
                         pipeline: bool = False, early_experts: Sequence[bool] | None = None,
                         gradient_outputs: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]] | None = None,
                         gradient_transposed: tuple[bool, bool] = (False, False),
                         consume_preactivation: bool = False,
                         w2_before_input_grad: bool = False, consume_derivative: bool = False) -> BackwardWork:
        """Allocate B outputs while reusing storage after each preceding reader.

        The returned output/readiness and two FP32 gradient tensors can be used
        to prepare token and owner returns before the shared phase admission.
        Standalone activation storage aliases dz. The transport pipeline keeps
        them independent so complete dW reductions can run after dX publication.
        By default da and dx share max(H, I) project columns. With derivative
        consumption, da uses I project columns and dx aliases phase-local dv.
        Arena slots are activation/project/down_gradient/gate_gradient; saved
        state and the two gradient outputs must not share those writable slots.
        For unclipped pipelined execution, explicit preactivation consumption
        computes A first, then overwrites the private saved Z with dZ.
        W2-first execution completes dA and a separate full W2 kernel before
        dZ and the consuming dX tail. Derivative consumption therefore needs
        only I columns of project scratch; dX then consumes the phase-local dY.
        """
        _load_native()
        self._check_input(dv, "backward derivative")
        self._check_input(state.x, "saved expert input")
        self._check_input(state.preact, "saved expert preactivation")
        if state.preact.shape[1] % 32 or dv.shape[1] != state.x.shape[1]:
            raise ValueError("saved preactivation and derivative do not describe the same expert MLP")
        if w2_before_input_grad and not pipeline:
            raise ValueError("W2-first ordering requires the backward pipeline")
        if consume_derivative and not w2_before_input_grad:
            raise ValueError("dY consumption requires every W2 reader to finish before dX")
        rows, hidden = state.x.shape
        intermediate = state.preact.shape[1] // 2
        consume_preactivation = bool(consume_preactivation and pipeline and self.swiglu_limit is None)
        dz = (state.preact if consume_preactivation else
              self._allocate(allocator, "activation", tuple(state.preact.shape), dv.dtype))
        activation = (self._allocate(allocator, "backward_activation", (rows, intermediate), dv.dtype)
                      if pipeline else dz.view(-1)[:rows * intermediate].view(rows, intermediate))
        columns = intermediate if consume_derivative else max(hidden, intermediate)
        storage = self._allocate(allocator, "project", (rows, columns), dv.dtype).view(-1)
        da = storage[:rows * intermediate].view(rows, intermediate)
        dx = dv if consume_derivative else storage[:rows * hidden].view(rows, hidden)
        _check_output_aliases(dz, dv, state.x, storage, *gate, *down,
                              *(() if consume_preactivation else (state.preact,)))
        _check_output_aliases(storage, dv, state.x, state.preact, *gate, *down)
        if pipeline:
            _check_output_aliases(activation, dz, dv, state.x, state.preact, storage, *gate, *down)
        da_work = self.prepare_linear(dv, tuple(weight.t() for weight in down), intermediate, output=da)
        if gradient_outputs is None:
            down_gradient = self._allocate(allocator, "down_gradient",
                                           (len(self.boundaries), intermediate, hidden), torch.float32)
            gate_gradient = self._allocate(allocator, "gate_gradient",
                                           (len(self.boundaries), hidden, 2 * intermediate), torch.float32)
            _check_output_aliases(down_gradient, gate_gradient, dv, state.x, state.preact, dz, storage, *gate, *down)
            _check_output_aliases(gate_gradient, dv, state.x, state.preact, dz, storage, *gate, *down)
            down_work = self.prepare_gradient(activation, dv, output=down_gradient)
            gate_work = self.prepare_gradient(state.x, dz, output=gate_gradient)
        else:
            if not pipeline:
                raise ValueError("mapped gradient accumulation requires the backward tail pipeline")
            gate_work = self.prepare_mapped_gradient(state.x, dz, gradient_outputs[0], gradient_transposed[0])
            down_work = self.prepare_mapped_gradient(activation, dv, gradient_outputs[1], gradient_transposed[1])
        dx_work = self.prepare_linear(dz, tuple(weight.t() for weight in gate), hidden, publish=True, output=dx)
        priorities, gradient_ready = None, None
        if pipeline:
            selected = tuple(early_experts) if early_experts is not None else (False,) * len(self.boundaries)
            if len(selected) != len(self.boundaries) or any(not isinstance(value, bool) for value in selected):
                raise ValueError("early gradient selection must contain one boolean per active expert")
            priorities = upload_metadata(torch.tensor(selected, dtype=torch.int32), self.device)
            flags = torch.zeros((dx_work.cores, 16), dtype=torch.int32, device=self.device)
            initialized = torch.npu.Event()
            initialized.record(torch.npu.current_stream(self.device))
            gradient_ready = ProducerFence(flags, initialized,
                                           None if dx_work.ready is None else dx_work.ready.poll_after)
        return BackwardWork(self, dv, state, da, dz, activation, dx, dx_work.ready,
                            down_work.result, gate_work.result, da_work, dx_work, down_work, gate_work,
                            pipeline=pipeline, early_experts=priorities, gradient_ready=gradient_ready,
                            consume_preactivation=consume_preactivation,
                            w2_before_input_grad=w2_before_input_grad, consume_derivative=consume_derivative)

    def run_mapped_gradient(self, work: GradientWork, priorities: torch.Tensor) -> None:
        """Complete all W2 tiles before dX may consume its private dY storage.

        The preceding dA kernel has already consumed the complete dispatched
        derivative. A separate compute-stream launch provides a full-core join;
        owner g receives FP32 additions and replica destinations are overwritten.
        """
        self._claim_unsubmitted_work(work)
        current = torch.npu.current_stream(self.device)
        current.wait_event(work.initialized)
        if work.cores:
            torch.ops.hyper_parallel.routed_experts_mapped_gradient(
                (work.x, work.dy, work.tasks, work.config, work.output_table, priorities), work.result, work.cores)

    def _run_backward_tail(self, work: BackwardWork) -> None:
        """Publish early dW and dX in one launch, then finish the deferred complete dW."""
        dx, gate, down = work.dx_work, work.gate_work, work.down_work
        for item in ((dx, gate) if work.w2_before_input_grad else (dx, gate, down)):
            self._claim_unsubmitted_work(item)
        current = torch.npu.current_stream(self.device)
        for item in (dx, gate, down):
            current.wait_event(item.initialized)
        current.wait_event(work.gradient_ready.initialized)
        if not dx.cores:
            return
        current.wait_event(dx.ready.admitted)
        dx.ready.poll_after.event.record(current)
        operands = (work.dz, dx.weight_table, dx.tasks, dx.config, work.state.x,
                    work.activation, work.dv, gate.tasks, gate.config, down.tasks, down.config, work.early_experts)
        extra = (work.consume_derivative,) if work.w2_before_input_grad else ()
        if gate.output_table is None:
            operation = (torch.ops.hyper_parallel.routed_experts_backward_tail_w2_first
                         if work.w2_before_input_grad else torch.ops.hyper_parallel.routed_experts_backward_tail)
            operation(
                dx.weights, operands, work.output, work.gate_gradient, work.down_gradient,
                dx.ready.flags, work.gradient_ready.flags, dx.transposed, dx.cores, *extra)
        else:
            operation = (torch.ops.hyper_parallel.routed_experts_backward_accumulate_w2_first
                         if work.w2_before_input_grad else torch.ops.hyper_parallel.routed_experts_backward_accumulate)
            operation(
                dx.weights, (*operands, gate.output_table, down.output_table), work.output,
                work.gate_gradient, work.down_gradient, dx.ready.flags, work.gradient_ready.flags,
                dx.transposed, gate.transposed, down.transposed, dx.cores, *extra)
        dx.ready.poll_after.queued = True

    @torch.no_grad()
    def run_backward(self, work: BackwardWork, publish: Callable[[str, torch.Tensor], None],
                     dependency: ExpertReadiness | None = None, *,
                     submit_return: Callable[[], None] | None = None) -> tuple:
        """Queue a return-producing pipeline or the standalone aliased backward.

        The publish callback may enqueue an asynchronous gradient return, but
        must restore the current compute stream before returning. submit_return
        queues token transport without joining it before dW has been submitted.
        """
        if work.pipeline and submit_return is None:
            raise ValueError("backward pipeline requires its prepared token return submission")
        self._claim_unsubmitted_work(work)
        self.run_linear(work.da_work, dependency)
        if work.pipeline:
            self._activate(work.state.preact, work.activation)
            if work.w2_before_input_grad:
                if work.down_work.output_table is None:
                    self.run_gradient(work.down_work)
                else:
                    self.run_mapped_gradient(work.down_work, work.early_experts)
            self._activation_gradient(work.da, work.state.preact, work.dz,
                                      consume_preactivation=work.consume_preactivation)
            self._run_backward_tail(work)
            publish("down", work.down_gradient)
            publish("gate", work.gate_gradient)
            submit_return()
            return work.output, work.ready
        self._activate(work.state.preact, work.activation)
        self.run_gradient(work.down_work)
        publish("down", work.down_gradient)
        self._activation_gradient(work.da, work.state.preact, work.dz)
        self.run_gradient(work.gate_work)
        publish("gate", work.gate_gradient)
        self.run_linear(work.dx_work)
        return work.output, work.ready


def combine(values: torch.Tensor, inverse_order: torch.Tensor, scores: torch.Tensor | None,
            tokens: int, top_k: int, *, output: torch.Tensor | None = None) -> torch.Tensor:
    """Sum top-k slots into BF16 output, retaining FP32 score products when supplied."""
    _load_native()
    if values.ndim != 2:
        raise ValueError("combine values must be a packed matrix")
    result = values.new_empty((tokens, values.shape[1])) if output is None else output
    _check_tensor(result, (tokens, values.shape[1]), values.dtype, values.device, "combine output")
    _check_output_aliases(result, values, inverse_order, *(() if scores is None else (scores,)))
    torch.ops.hyper_parallel.routed_experts_combine(values, inverse_order, scores, result, top_k, 48)
    return result


def combine_gradient(dy: torch.Tensor, values: torch.Tensor, order: torch.Tensor, scores: torch.Tensor, *,
                     outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
                     consume_values: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BF16 value and score-matched gradients into admitted outputs.

    consume_values requires exclusive ownership after the saved expert outputs'
    final read. Each row loads its values before overwriting that exact row with
    weighted dY; no other row reads it. The ordinary API rejects every alias.
    """
    _load_native()
    if outputs is None:
        dv = values if consume_values else torch.empty_like(values)
        ds = torch.empty_like(scores)
    else:
        if len(outputs) != 2:
            raise ValueError("combine gradient requires exactly two prepared outputs")
        dv, ds = outputs
    _check_tensor(dv, tuple(values.shape), values.dtype, values.device, "combine value gradient")
    _check_tensor(ds, tuple(scores.shape), scores.dtype, values.device, "combine score gradient")
    if consume_values:
        if dv.data_ptr() != values.data_ptr() or dv.stride() != values.stride():
            raise ValueError("consumed combine values must be the exact dV output")
        _check_output_aliases(dv, dy, order, scores, ds)
    else:
        _check_output_aliases(dv, dy, values, order, scores, ds)
    _check_output_aliases(ds, dy, values, order, scores, dv)
    if consume_values:
        torch.ops.hyper_parallel.routed_experts_combine_backward_consume(dy, values, order, scores, ds, 48)
    else:
        torch.ops.hyper_parallel.routed_experts_combine_backward(dy, values, order, scores, dv, ds, 48)
    return dv, ds
