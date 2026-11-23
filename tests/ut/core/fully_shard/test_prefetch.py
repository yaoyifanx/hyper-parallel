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
"""Execution order, depth and demand-loaded subgraph boundaries."""
import unittest

from torch import nn

from hyper_parallel.core.fully_shard.prefetch import build_execution_prefetch_plan


class TestExecutionPrefetch(unittest.TestCase):
    """Expert and router backward order must follow the actual forward graph."""

    def setUp(self) -> None:
        """Build explicit execution groups with an unrelated demand-loaded unit."""
        self.model = nn.Module()
        self.model.embedding = nn.Linear(2, 2)
        self.model.blocks = nn.ModuleList([nn.Module() for _ in range(3)])
        for block in self.model.blocks:
            # Deliberately register in a different order from execution.
            block.experts, block.router = nn.Linear(2, 2), nn.Linear(2, 2)
        self.groups = {block: [block, block.router, block.experts] for block in self.model.blocks}
        self.groups[self.model.embedding] = [self.model.embedding]

    def test_depth_two_and_recompute(self):
        """A block prefetches two future units; experts backward prefetches router then next block."""
        plan = build_execution_prefetch_plan(self.model, self.model.blocks, self.groups,
                                             forward_depth=2, backward_depth=2, recompute_forward=True)
        self.assertEqual(plan["units"]["blocks.0"]["forward"], ["blocks.1", "blocks.1.router"])
        self.assertEqual(plan["units"]["blocks.2.experts"]["backward"], ["blocks.2.router", "blocks.1"])
        self.assertEqual(plan["units"]["blocks.1.router"]["forward"], [])
        self.assertEqual(plan["units"]["blocks.0.router"]["backward"], [])
        self.assertEqual(plan["demand_loaded_units"], ["embedding"])
        self.assertEqual(plan["inflight"]["fwd_max"], 2)

    def test_invalid_and_duplicate_units(self):
        """Reject ambiguous order rather than silently issuing extra gathers."""
        with self.assertRaises(ValueError):
            build_execution_prefetch_plan(self.model, [self.model.blocks[0]] * 2, self.groups,
                                           forward_depth=2, backward_depth=2, recompute_forward=True)
        with self.assertRaises(ValueError):
            build_execution_prefetch_plan(self.model, [nn.Module()], self.groups,
                                           forward_depth=2, backward_depth=2, recompute_forward=True)

    def test_zero_depth(self):
        """Explicit depth zero preserves demand loading."""
        plan = build_execution_prefetch_plan(self.model, self.model.blocks, self.groups,
                                             forward_depth=0, backward_depth=0, recompute_forward=False)
        self.assertTrue(all(not targets[direction] for targets in plan["units"].values()
                            for direction in ("forward", "backward")))
