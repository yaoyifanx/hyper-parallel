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
"""External EP ownership participates in RNG without changing FSDP parameters."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Partial, Replicate, Shard
from hyper_parallel.core.dtensor.random import OffsetBasedRNGTracker
from hyper_parallel.core.dtensor.random_init import RNGLayout, get_rng_layout, module_rng_layouts
from hyper_parallel.core.shard._op_dispatch import OpDispatcher, get_dtensor_dispatch


def fake_mesh(shape: tuple[int, ...], coordinate: tuple[int, ...]) -> SimpleNamespace:
    """Provide metadata only; unit tests do not initialize process groups."""
    return SimpleNamespace(ndim=len(shape), mesh_shape=shape, get_coordinate=lambda: coordinate,
                           size=lambda axis: shape[axis])


class TestModuleRNGLayouts(unittest.TestCase):
    """Scoped overrides retain identities and handle every named parameter."""

    def setUp(self) -> None:
        """Create CPU parameters and a metadata-only EP16 layout."""
        self.model = nn.Module()
        self.model.register_parameter("first", nn.Parameter(torch.empty(8, 6)))
        self.model.register_parameter("second", nn.Parameter(torch.empty(8, 6)))
        self.layout = RNGLayout(fake_mesh((16, 1), (0, 0)), (Shard(0), Shard(1)), (128, 6))

    def test_all_parameters_and_exception_restore(self):
        """The second parameter is bound, and exceptions release every binding."""
        first, second = self.model.first, self.model.second
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with module_rng_layouts(self.model, {"first": self.layout, "second": self.layout}):
                self.assertIs(get_rng_layout(second), self.layout)
                self.assertIs(self.model.first, first)
                self.assertTrue(get_dtensor_dispatch())
                raise RuntimeError("injected")
        self.assertIsNone(get_rng_layout(first))
        self.assertIsNone(get_rng_layout(second))
        self.assertIs(self.model.second, second)

    def test_nested_scope_restores_outer_layout(self):
        """A nested context cannot leave its layout in later training calls."""
        other = RNGLayout(fake_mesh((8, 1), (0, 0)), (Shard(0), Shard(1)), (64, 6))
        with module_rng_layouts(self.model, {"first": self.layout}):
            with module_rng_layouts(self.model, {"first": other}):
                self.assertIs(get_rng_layout(self.model.first), other)
            self.assertIs(get_rng_layout(self.model.first), self.layout)
            self.assertIsNone(get_rng_layout(self.model.second))
        self.assertIsNone(get_rng_layout(self.model.first))

    def test_invalid_layouts_fail_before_binding(self):
        """Local storage mismatch and partial ownership must not be accepted."""
        cases = [RNGLayout(self.layout.device_mesh, (Shard(0), Shard(1)), (64, 6)),
                 RNGLayout(self.layout.device_mesh, (Partial(), Shard(1)), (8, 6)),
                 RNGLayout(self.layout.device_mesh, (Shard(0),), (128, 6))]
        for layout in cases:
            with self.subTest(layout=layout):
                with self.assertRaises(ValueError):
                    with module_rng_layouts(self.model, {"first": layout}):
                        self.fail(f"invalid RNG layout entered its context: layout={layout}")
                self.assertIsNone(get_rng_layout(self.model.first))


class TestRNGLayoutDispatch(unittest.TestCase):
    """The native random dispatcher receives EP metadata without a new tensor."""

    def test_dispatch_uses_override_or_original_layout(self):
        """The scoped full shape replaces only the RNG tracker's three arguments."""
        parameter = MagicMock(spec=DTensor)
        parameter.to_local.return_value = torch.empty(8, 6)
        parameter.shape = (8, 6)
        parameter.placements = (Shard(1),)
        parameter.device_mesh = fake_mesh((1,), (0,))
        override = RNGLayout(fake_mesh((16, 1), (0, 0)), (Shard(0), Shard(1)), (128, 6))
        for layout in (None, override):
            with self.subTest(override=layout is not None):
                dispatcher = object.__new__(OpDispatcher)
                tracker = MagicMock()
                tracker.distribute_region_enabled = True
                tracker._distribute_region.side_effect = lambda **kwargs: nullcontext()
                dispatcher._rng_tracker = tracker
                kernel = MagicMock(return_value=parameter.to_local())
                with patch("hyper_parallel.core.shard._op_dispatch.get_rng_layout", return_value=layout):
                    result = dispatcher._dispatch_random_op("normal_", kernel, (parameter,), {"std": 0.02})
                self.assertIs(result, parameter)
                tracker._distribute_region.assert_called_once_with(
                    device_mesh=parameter.device_mesh if layout is None else layout.device_mesh,
                    placements=parameter.placements if layout is None else layout.placements,
                    global_shape=parameter.shape if layout is None else layout.global_shape,
                    generator=None)
                kernel.assert_called_once_with(parameter.to_local(), std=0.02)

    def test_ep16_offsets_and_replicated_shard_offsets(self):
        """Distinct EP shards separate their streams; replicated shards agree."""
        tracker = object.__new__(OffsetBasedRNGTracker)
        shape = (128, 2816, 4096)
        size = 8 * 2816 * 4096
        offsets = [tracker.compute_offset_incr(fake_mesh((16, 1), (rank, 0)),
                                              (Shard(0), Shard(1)), shape) for rank in range(16)]
        expected = [rank * size for rank in range(16)]
        self.assertEqual(offsets, expected, msg=f"offsets={offsets}, expected={expected}")
        replicas = [[tracker.compute_offset_incr(fake_mesh((2, 4, 2), (replica, ep, fsdp)),
                                                (Replicate(), Shard(0), Shard(1)), shape)
                     for ep in range(4) for fsdp in range(2)] for replica in range(2)]
        self.assertEqual(replicas[0], replicas[1], msg=f"replica0={replicas[0]}, replica1={replicas[1]}")
        self.assertEqual(len(set(replicas[0])), 8, msg=f"offsets={replicas[0]}, expected_unique=8")
