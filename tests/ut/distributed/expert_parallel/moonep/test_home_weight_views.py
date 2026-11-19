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
"""Borrowed home matrix ownership and remote-only materialization contracts."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.numerics import (
    GroupedWeightViews, _grouped_parts, grouped_values,
)
from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPRuntime


class TestHomeWeightViews(unittest.TestCase):
    """Retain the established grouped GEMM row order while borrowing home storage."""

    def setUp(self) -> None:
        """Prepare transposed home storage and one remote contribution."""
        self.home = (torch.randn(2, 8, 4).bfloat16().transpose(1, 2),
                     torch.randn(2, 4, 4).bfloat16().transpose(1, 2))
        self.remote = (torch.randn(1, 4, 8).bfloat16(), torch.randn(1, 4, 4).bfloat16())
        runtime = MoonEPRuntime.__new__(MoonEPRuntime)
        runtime.stream = None
        runtime.slots, runtime.weight_leases, runtime.weight_workspace = [], {}, {}
        runtime.weight_workspace_busy = runtime.release_w_workspace = False
        runtime.home_weight_views = True
        runtime.statistics = Counter()
        executor = SimpleNamespace(read_weights=lambda: self.home, replica_capacity=2)
        plan = SimpleNamespace(weight_recv=(0, 1), weight_send=(0, 0))
        runtime.executions = {"key": (executor, plan)}
        runtime._exchange_weights = Mock(return_value=(*self.remote, []))
        self.runtime = runtime

    def test_materialization_borrows_home_and_copies_only_remote(self):
        """Remote capacity must not include another complete home allocation."""
        lease = self.runtime._prepare_weight_views("key")
        for home, remote, grouped, table in zip(self.home, self.remote,
                                               (lease.gate_up, lease.down),
                                               (lease.slot.gate_up, lease.slot.down)):
            for index in range(2):
                self.assertEqual(grouped.matrices[index].data_ptr(), home[index].data_ptr())
            torch.testing.assert_close(grouped.matrices[2], remote[0], rtol=0, atol=0)
            self.assertEqual(table.shape[0], 2)
            self.assertEqual(table.untyped_storage().nbytes(), 2 * remote[0].numel() * 2)
            self.assertEqual(table.stride(-2), home.stride(-2))

    def test_grouped_forward_and_transposed_backward_match_stacked_reference(self):
        """Zero-token groups and matrix transposes must preserve group boundaries."""
        lease = self.runtime._prepare_weight_views("key")
        groups = torch.tensor([3, 3, 8])
        x = torch.randn(8, 4).bfloat16()
        matrices = lease.gate_up.matrices
        expected = grouped_values(x, torch.stack(matrices), groups)
        actual = grouped_values(x, lease.gate_up, groups)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        transposed = lease.gate_up.transpose(-1, -2)
        torch.testing.assert_close(grouped_values(actual, transposed, groups),
                                   grouped_values(actual, torch.stack(matrices).transpose(1, 2), groups),
                                   rtol=0, atol=0)

    def test_record_stream_deduplicates_shared_home_and_replica_storage(self):
        """Several home experts must retain their source storage only once."""
        lease = self.runtime._prepare_weight_views("key")
        with patch.object(torch.Tensor, "record_stream") as record:
            lease.gate_up.record_stream(object())
        self.assertEqual(record.call_count, 2)

    def test_workspace_retires_after_materialization(self):
        """Only final remote weights remain necessary after unpack submission."""
        self.runtime.release_w_workspace = True
        self.runtime.weight_workspace = {"send": torch.empty(16), "recv": torch.empty(32)}
        lease = self.runtime._prepare_weight_views("key")
        self.assertEqual(self.runtime.weight_workspace, {})
        torch.testing.assert_close(lease.gate_up.matrices[2], self.remote[0][0], rtol=0, atol=0)

    def test_release_mode_does_not_restore_historical_peak_workspace(self):
        """A short request allocates its own size after retiring a longer request."""
        self.runtime.release_w_workspace = True
        self.runtime.weight_workspace_demand = {"send": 64}
        actual = self.runtime._borrow_weight_workspace("send", torch.empty(1), 16, 128)
        self.assertEqual(actual.untyped_storage().nbytes(), 16 * 4)

    def test_invalid_matrix_lists_are_rejected(self):
        """The installed grouped GEMM requires uniform matrices and at most 128 entries."""
        with self.assertRaisesRegex(ValueError, "128"):
            GroupedWeightViews(tuple(torch.ones(2, 2) for _ in range(129)))
        with self.assertRaisesRegex(ValueError, "one shape"):
            GroupedWeightViews((torch.ones(2, 2), torch.ones(2, 3)))

    def test_batched_parts_borrow_home_without_building_expert_matrix_list(self):
        """Freeze only two batch descriptors and their CPU-known row partition."""
        self.runtime.home_weight_mode = "2"
        self.runtime.executions["key"][1].home_count = 2
        self.runtime.executions["key"][1].group_counts = (3, 0, 5)
        lease = self.runtime._prepare_weight_views("key")
        self.assertEqual(lease.gate_up.matrices, ())
        self.assertEqual(lease.gate_up.rows_per_part, (3, 5))
        self.assertIs(lease.gate_up.parts[0], self.home[0])
        torch.testing.assert_close(lease.gate_up.parts[1], self.remote[0], rtol=0, atol=0)

    def test_batched_parts_preserve_zero_groups_and_output_row_order(self):
        """Remote cumulative boundaries must be relative to its sliced input."""
        boundaries = torch.tensor([3, 3, 8])
        weight = GroupedWeightViews(parts=(self.home[0], self.remote[0]), rows_per_part=(3, 5))
        x = torch.randn(8, 4).bfloat16()

        def gemm(**kwargs: object) -> list[torch.Tensor]:
            """Evaluate each requested batch independently on CPU."""
            return [grouped_values(kwargs["x"][0], kwargs["weight"][0], kwargs["group_list"])]

        operator = Mock(side_effect=gemm)
        actual = _grouped_parts(x, weight, boundaries, operator)
        expected = grouped_values(x, weight, boundaries)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(operator.call_count, 2)
        torch.testing.assert_close(operator.call_args_list[1].kwargs["group_list"], torch.tensor([5]))
        transposed = weight.transpose(1, 2)
        torch.testing.assert_close(grouped_values(actual, transposed, boundaries),
                                   grouped_values(actual, torch.cat((self.home[0], self.remote[0])).transpose(1, 2),
                                                  boundaries), rtol=0, atol=0)

    def test_one_active_part_returns_gemm_output_without_an_extra_copy(self):
        """No-migration and remote-only plans keep the ordinary one-GMM path."""
        x = torch.randn(5, 4).bfloat16()
        for counts in ((5, 0), (0, 5)):
            weight = GroupedWeightViews(parts=(self.home[0], self.remote[0]), rows_per_part=counts)
            boundaries = torch.tensor([5, 5, 5] if counts[0] else [0, 0, 5])
            result = torch.randn(5, 8).bfloat16()
            operator = Mock(return_value=[result])
            self.assertIs(_grouped_parts(x, weight, boundaries, operator), result)
            operator.assert_called_once()

    def test_parts_reject_uncovered_input_before_submitting_gemm(self):
        """Incorrect row metadata must fail before touching any output."""
        weight = GroupedWeightViews(parts=(self.home[0], self.remote[0]), rows_per_part=(3, 5))
        operator = Mock()
        with self.assertRaisesRegex(ValueError, "do not cover"):
            _grouped_parts(torch.randn(7, 4), weight, torch.tensor([3, 3, 8]), operator)
        operator.assert_not_called()
