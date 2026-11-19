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
"""Capacity-aware home layout ownership and lazy replica storage."""
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import ReplicaPool


class TestCompactWeights(unittest.TestCase):
    """Do not allocate speculative maximum replica tables before planning."""

    def test_original_route_defers_home_layout_until_plan_exists(self):
        """The early layout callback precedes publication of the execution plan."""
        runtime = MoonEPRuntime.__new__(MoonEPRuntime)
        runtime.compact_weight_slots = True
        runtime.executions = {}
        runtime._acquire_home_slot = Mock(side_effect=AssertionError("plan not published"))
        weights = (torch.empty(2, 3, 4).transpose(1, 2), torch.empty(2, 4, 3).transpose(1, 2))
        runtime.prepare_home_layout(SimpleNamespace(read_weights=lambda: weights), "key")
        runtime._acquire_home_slot.assert_not_called()

    def test_home_layout_aliases_final_table(self):
        """A transposed home parameter needs one final-table copy, no second home scratch."""
        runtime = MoonEPRuntime.__new__(MoonEPRuntime)
        runtime.compact_weight_slots = True
        runtime.executions = {"key": (None, SimpleNamespace(group_counts=(1, 2, 3)))}
        runtime.executors = [SimpleNamespace(weight_group=None)]
        runtime.home_layout_slots, runtime.slots = {}, []
        runtime.capacity = 16
        runtime.stream = runtime.home_layout_stream = None
        runtime.statistics = Counter()
        weights = (torch.randn(2, 3, 4).transpose(1, 2), torch.randn(2, 4, 3).transpose(1, 2))
        runtime.prepare_home_layout(SimpleNamespace(read_weights=lambda: weights), "key")
        slot = runtime.slots[0]
        self.assertEqual(slot.gate_up.shape[0], 4)
        for final, home, source in zip((slot.gate_up, slot.down), (slot.home_gate_up, slot.home_down), weights):
            self.assertEqual(final.untyped_storage().data_ptr(), home.untyped_storage().data_ptr())
            torch.testing.assert_close(home, source)

    def test_lazy_replica_slots_and_retirement(self):
        """Replica slots retain only observed capacity across retirement/restoration."""
        with patch.dict("os.environ", {"HP_MOONEP_COMPACT_WEIGHTS": "1"}):
            pool = ReplicaPool(torch.device("cpu"), torch.bfloat16, slot_count=2)
        pool.allocate(8, 8, ((16, 32), (32, 16)), (True, False))
        self.assertEqual(pool.reserved_bytes, 0)
        pool.ensure_capacity(pool.slots[0], 1)
        self.assertEqual(pool.slots[0].capacity, 4)
        expected = 4 * (16 * 32 + 32 * 16) * 2
        self.assertEqual(pool.reserved_bytes, expected)
        pool.retire_slots()
        self.assertEqual(pool.reserved_bytes, 0)
        pool.restore_slots()
        self.assertEqual(pool.reserved_bytes, expected)
