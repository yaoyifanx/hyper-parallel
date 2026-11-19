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
"""Scoped logical RNG layouts for parameters sliced before FSDP wrapping."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.placement_types import Placement
from hyper_parallel.core.dtensor.random import local_shard_size_and_offset


@dataclass(frozen=True)
class RNGLayout:
    """Full logical ownership used only to select a parameter's random stream."""

    device_mesh: DeviceMesh
    placements: tuple[Placement, ...]
    global_shape: tuple[int, ...]


_RNG_LAYOUTS: ContextVar[Optional[dict[int, RNGLayout]]] = ContextVar("parameter_rng_layouts", default=None)


def get_rng_layout(tensor: torch.Tensor) -> Optional[RNGLayout]:
    """Return an active initialization layout, or None for normal dispatch."""
    layouts = _RNG_LAYOUTS.get()
    return None if layouts is None else layouts.get(id(tensor))


def _validate_layout(parameter: torch.Tensor, layout: RNGLayout) -> None:
    """Reject a logical layout that does not describe the existing local storage."""
    mesh = layout.device_mesh
    if len(layout.placements) != mesh.ndim or not layout.global_shape:
        raise ValueError("RNG layout must provide one placement per mesh dimension and a positive tensor rank")
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        raise ValueError("parameter RNG layout must include the current rank")
    expected = list(layout.global_shape)
    if any(extent < 0 for extent in expected):
        raise ValueError("parameter RNG global shape cannot contain negative dimensions")
    for axis, placement in enumerate(layout.placements):
        if placement.is_shard():
            dim = placement.dim
            if not -len(expected) <= dim < len(expected):
                raise ValueError("RNG shard dimension is outside the parameter rank")
            expected[dim], _ = local_shard_size_and_offset(expected[dim], mesh.size(axis), coordinate[axis])
        elif not placement.is_replicate():
            raise ValueError("parameter RNG layouts support only shard and replicate placements")
    local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
    if tuple(local.shape) != tuple(expected):
        raise ValueError(f"RNG layout local shape mismatch: expected {tuple(expected)}, got {tuple(local.shape)}")


@contextmanager
def module_rng_layouts(model: nn.Module, layouts: Mapping[str, RNGLayout]) -> Iterator[None]:
    """Bind full logical RNG layouts to current module parameters during initialization.

    Args:
        model: Materialized module whose parameters are about to be initialized.
        layouts: Parameter names and complete layouts, including external EP slices.
            Every named parameter is bound independently; unlisted tensors retain
            their original RNG behavior. Nested scopes restore earlier bindings.

    Note:
        The scope changes RNG metadata only. It preserves parameter identities,
        storage, training layouts and DTensor dispatch. Enter it after to_empty(),
        since materialization may replace Parameter objects.
    """
    bindings = dict(_RNG_LAYOUTS.get() or {})
    parameters = []
    for name, layout in layouts.items():
        parameter = model.get_parameter(name)
        _validate_layout(parameter, layout)
        parameters.append(parameter)
        bindings[id(parameter)] = layout
    token = _RNG_LAYOUTS.set(bindings)
    try:
        yield
    finally:
        _RNG_LAYOUTS.reset(token)
