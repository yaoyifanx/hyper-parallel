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
"""Maximum-capacity weight storage for actually borrowed logical slots."""
from collections import Counter
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPExecutor, MoonEPRuntime, configure_moonep
from hyper_parallel.distributed.expert_parallel.moonep.fused.executor import SparseMoonEPExecutor
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import ReplicaPool


class TestMaximumWeightStorage(unittest.TestCase):
    """Use maximum capacity per physical slot without allocating unused slots."""

    def test_original_route_defers_home_layout_until_plan_exists(self):
        """The early layout callback precedes publication of the execution plan."""
        runtime = MoonEPRuntime.__new__(MoonEPRuntime)
        runtime.executions = {}
        runtime._acquire_home_slot = Mock(side_effect=AssertionError("plan not published"))
        weights = (torch.empty(2, 3, 4).transpose(1, 2), torch.empty(2, 4, 3).transpose(1, 2))
        runtime.prepare_home_layout(SimpleNamespace(read_weights=lambda: weights), "key")
        runtime._acquire_home_slot.assert_not_called()

    def test_home_layout_aliases_final_table(self):
        """A transposed home parameter needs one final-table copy, no second home scratch."""
        runtime = MoonEPRuntime.__new__(MoonEPRuntime)
        runtime.executions = {"key": (None, SimpleNamespace(group_counts=(1, 2, 3)))}
        runtime.executors = [SimpleNamespace(weight_group=None)]
        runtime.home_layout_slots, runtime.slots = {}, []
        runtime.capacity = 16
        runtime.stream = runtime.home_layout_stream = None
        runtime.statistics = Counter()
        weights = (torch.randn(2, 3, 4).transpose(1, 2), torch.randn(2, 4, 3).transpose(1, 2))
        runtime.prepare_home_layout(SimpleNamespace(read_weights=lambda: weights), "key")
        slot = runtime.slots[0]
        self.assertEqual(slot.gate_up.shape[0], 16)
        for final, home, source in zip((slot.gate_up, slot.down), (slot.home_gate_up, slot.home_down), weights):
            self.assertEqual(final.untyped_storage().data_ptr(), home.untyped_storage().data_ptr())
            torch.testing.assert_close(home, source)

    def test_maximum_capacity_only_for_used_slot_and_restoration(self):
        """Clearing/restoring retains maximum capacity only for a previously used slot."""
        pool = ReplicaPool(torch.device("cpu"), torch.bfloat16, slot_count=3)
        pool.allocate(8, 8, ((16, 32), (32, 16)), (True, False))
        self.assertEqual(pool.reserved_bytes, 0)
        pool.ensure_storage(pool.slots[0], 1)
        self.assertTrue(pool.slots[0].initialized)
        self.assertEqual(pool.slots[0].matrices["gate"].shape[0], 8)
        expected = 8 * (16 * 32 + 32 * 16) * 2
        self.assertEqual(pool.reserved_bytes, expected)
        self.assertEqual([slot.initialized for slot in pool.slots], [True, False, False])
        pool.retire_slots()
        self.assertEqual(pool.reserved_bytes, 0)
        pool.restore_slots()
        self.assertEqual(pool.reserved_bytes, expected)


    def test_legacy_environment_does_not_change_policy(self):
        """Old launch shells cannot override maximum capacity on first use."""
        self.enterContext(patch("hyper_parallel.distributed.expert_parallel.moonep.fused.executor.dist.get_rank",
                                return_value=0))
        for constructor in (MoonEPExecutor, SparseMoonEPExecutor):
            for value in ("0", "1", "invalid"):
                with self.subTest(backend=constructor.__name__, value=value):
                    owner = torch.nn.Module()
                    owner._moonep_executor = constructor(
                        owner, Mock(), ep_group=None, ep_size=1, num_experts=4, top_k=2)
                    with patch.dict("os.environ", {"HP_MOONEP_COMPACT_WEIGHTS": value}):
                        configured = configure_moonep(owner)
                        pool = ReplicaPool(torch.device("cpu"), torch.bfloat16)
                    self.assertIs(owner._moonep_executor.runtime, configured)
                    pool.allocate(8, 8, ((16, 32), (32, 16)), (True, False))
                    self.assertEqual(pool.reserved_bytes, 0)
                    pool.ensure_storage(pool.slots[0], 1)
                    self.assertEqual(pool.slots[0].matrices["gate"].shape[0], 8)


    def test_small_then_large_demand_reuses_the_same_maximum_storage(self):
        """A later replica increase cannot introduce an additional allocation or cast."""
        pool = ReplicaPool(torch.device("cpu"), torch.bfloat16, slot_count=3)
        pool.allocate(8, 6, ((16, 32), (32, 16)), (True, False))
        slot = pool.slots[0]
        pool.ensure_storage(slot, 1)
        self.assertTrue(slot.initialized)
        self.assertEqual(slot.matrices["gate"].shape[0], 6)
        pointers = tuple(value.data_ptr() for value in slot.matrices.values())
        with patch.object(torch, "empty", side_effect=AssertionError("must reuse storage")):
            pool.ensure_storage(slot, 5)
        self.assertEqual(tuple(value.data_ptr() for value in slot.matrices.values()), pointers)
        self.assertTrue(all(not unused.matrices and not unused.initialized for unused in pool.slots[1:]))
        for demand in (-1, 7, True):
            with self.subTest(demand=demand):
                with self.assertRaisesRegex(ValueError, "admitted capacity"):
                    pool.ensure_storage(slot, demand)
        slot.in_use = True
        with self.assertRaisesRegex(ValueError, "idle slot"):
            pool.ensure_storage(slot, 1)

    def test_zero_remote_demand_does_not_allocate_storage(self):
        """A home-only invocation needs a logical lease but no physical replica bytes."""
        pool = ReplicaPool(torch.device("cpu"), torch.bfloat16, slot_count=3)
        pool.allocate(8, 8, ((16, 32), (32, 16)), (True, False))
        with patch.object(torch, "empty", side_effect=AssertionError("unused replica storage")):
            pool.ensure_storage(pool.slots[0], 0)
            pool.retire_slots()
            pool.restore_slots()
        self.assertEqual(pool.reserved_bytes, 0)
        self.assertTrue(all(not slot.matrices for slot in pool.slots))

    def test_partial_allocation_failure_keeps_unused_slot_empty(self):
        """A failed first use releases its partial maximum allocation before admission."""
        pool = ReplicaPool(torch.device("cpu"), torch.bfloat16)
        pool.allocate(8, 8, ((16, 32), (32, 16)), (True, False))
        original = torch.empty
        calls = []
        references = []

        def allocate(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise MemoryError("second matrix")
            value = original(*args, **kwargs)
            references.append(weakref.ref(value))
            return value

        with patch.object(torch, "empty", side_effect=allocate):
            with self.assertRaisesRegex(MemoryError, "second matrix"):
                pool.ensure_storage(pool.slots[0], 1)
        self.assertEqual(pool.reserved_bytes, 0)
        self.assertFalse(pool.slots[0].initialized)
        self.assertFalse(pool.slots[0].matrices)
        self.assertTrue(all(reference() is None for reference in references))
