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
"""Check expert identities and wire ownership with host-topology NF planning."""

import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep.layout import route_geometry
from hyper_parallel.distributed.expert_parallel.moonep.planning import (
    compile_host_plan, compile_plan, prepare_entry_order, select_allocation,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestHostExecutionPlan(unittest.TestCase):
    """Exercise empty sources, uneven routing, fixed homes and remote expert slots."""

    def test_expert_identities_across_variable_sources(self) -> None:
        """Exchange actual source expert labels and verify every destination group."""
        topologies = ((1, 4, [0]), (2, 8, [23, 37]), (4, 16, [0, 17, 23, 41]),
                      (16, 128, [index * 7 for index in range(16)]))
        for ranks, experts, rows in topologies:
            with self.subTest(ranks=ranks, experts=experts):
                self._check_topology(ranks, experts, rows)

    def _check_topology(self, ranks: int, experts: int, rows: list[int]) -> None:
        """Build real peer slices and compare grouped expert identities independently."""
        torch.manual_seed(416 + ranks)
        weights = torch.ones(experts)
        weights[0] = experts
        choices = [torch.multinomial(weights, count, replacement=True)
                   if count else torch.empty(0, dtype=torch.long) for count in rows]
        counts = torch.stack([torch.bincount(ids, minlength=experts) for ids in choices])
        counts, allocation, fallback = select_allocation(counts, experts // ranks)
        plans, sends = [], []
        for rank, ids in enumerate(choices):
            entry = prepare_entry_order(ids, experts)
            reference = compile_plan(counts, allocation, ids, rank,
                                     replica_capacity=experts // ranks, entry_order=entry)
            plan = compile_host_plan(counts, allocation, ids, rank, replica_capacity=experts // ranks,
                                     entry_order=entry, planner_fallback=fallback)
            for field in ("dispatch_order", "unpermute_indices", "sort_order", "boundaries", "send_ids"):
                torch.testing.assert_close(getattr(reference, field), getattr(plan, field), rtol=0, atol=0)
            for field in ("group_counts", "token_send", "token_recv", "weight_send", "weight_recv",
                          "send_slots", "metrics"):
                self.assertEqual(getattr(reference, field), getattr(plan, field))
            self.assertEqual(plan.planner_fallback, fallback)
            restored = plan.dispatch_order[plan.unpermute_indices.long()]
            torch.testing.assert_close(restored, torch.arange(ids.numel()), rtol=0, atol=0)
            sends.append(torch.split(ids[plan.dispatch_order], tuple(plan.token_send)))
            plans.append(plan)
        for target, plan in enumerate(plans):
            received = torch.cat([sends[source][target] for source in range(ranks)])
            actual = received[plan.sort_order]
            geometry = route_geometry(counts, allocation, target, min(experts - experts // ranks, experts // ranks))
            active = geometry.active[:len(plan.group_counts)]
            expected = torch.repeat_interleave(active, torch.tensor(plan.group_counts))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
