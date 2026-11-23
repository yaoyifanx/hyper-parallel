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
"""CPU payload oracle for native FD with NF expert ordering."""

from collections import Counter
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hyper_parallel.distributed.expert_parallel.moonep import runtime, device_dispatch

from hyper_parallel.distributed.expert_parallel.moonep.planning import (
    select_allocation,
    compile_host_plan,
    prepare_entry_order,
)
from hyper_parallel.distributed.expert_parallel.moonep.device_dispatch import compile_device_forward
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestDeviceDispatchRows(unittest.TestCase):
    """Prove native rows preserve the existing NF GMM home-first ownership."""

    def test_original_token_identities_and_home_first_order(self) -> None:
        """Simulate peer payloads and verify every token identity and destination."""
        for ranks, experts, top_k, token_counts in [(1, 4, 2, [0]), (2, 8, 2, [7, 11]), (4, 16, 3, [0, 7, 13, 19]),
                                                    (16, 128, 8, [1 + i for i in range(16)])]:
            torch.manual_seed(714 + ranks)
            ids = [torch.randn(tokens, experts).topk(top_k).indices.reshape(-1) for tokens in token_counts]
            counts = torch.stack([torch.bincount(v, minlength=experts) for v in ids])
            counts, allocation, fallback = select_allocation(counts, experts // ranks)
            plans = []
            native = []
            wire = []
            actual = []
            for rank, choices in enumerate(ids):
                entry = prepare_entry_order(choices, experts)
                p = compile_host_plan(counts,
                                      allocation,
                                      choices,
                                      rank,
                                      replica_capacity=experts // ranks,
                                      entry_order=entry,
                                      planner_fallback=fallback)
                d = compile_device_forward(counts, allocation, rank, entry, top_k)
                d.route.validate()
                expected_identity = torch.equal(d.target_rows, torch.arange(d.route.received_rows))
                self.assertEqual(d.target_identity, expected_identity)
                payload = torch.arange(d.route.received_rows * 3).reshape(d.route.received_rows, 3).float()
                engine = device_dispatch.DeviceDispatcher()
                with mock.patch.object(device_dispatch, 'prepare_device_forward'), mock.patch.object(
                        device_dispatch, 'submit_prepared', return_value=SimpleNamespace(wait=lambda: payload)):
                    restored = engine.forward(payload, d, counted=True, ratio=.75)
                torch.testing.assert_close(restored, payload[d.target_rows], rtol=0, atol=0)
                if d.target_identity and payload.numel():
                    self.assertEqual(restored.data_ptr(), payload.data_ptr())
                hidden = torch.arange(token_counts[rank]) + rank * 1000
                wire.append(torch.split(hidden[p.dispatch_order // top_k], tuple(p.token_send)))
                plans.append(p)
                native.append(d)
                actual.append(torch.empty(d.route.received_rows, dtype=torch.long))
            for source, d in enumerate(native):
                hidden = torch.arange(token_counts[source]) + source * 1000
                for edge in d.route.outgoing:
                    actual[edge.target][edge.target_row:edge.target_row +
                                        edge.rows] = hidden[d.source_rows[edge.source_row:edge.source_row + edge.rows]]
            for target, p in enumerate(plans):
                expected = torch.cat([wire[source][target] for source in range(ranks)])[p.sort_order]
                torch.testing.assert_close(actual[target][native[target].target_rows], expected, rtol=0, atol=0)

    def test_device_counted_forward_does_not_read_host_savings(self) -> None:
        """Run the public NF call with device counters and ordinary combine semantics."""
        counts = torch.tensor([[2, 2]], dtype=torch.long)
        order = torch.tensor([0, 2, 1, 3])
        device_plan = compile_device_forward(counts, counts.T, 0, order, 2)
        plan = SimpleNamespace(dispatch_order=order, token_transport=device_plan)
        completed = []
        lifecycle = SimpleNamespace(executions={}, statistics=Counter(),
                                    forward_key=lambda *args: 'key', complete=completed.append)
        weights = (torch.zeros(1), torch.zeros(1))
        executor = SimpleNamespace(runtime=lifecycle, top_k=2, chunked=None)
        lifecycle.executions['key'] = executor, plan
        hidden = torch.zeros(2, 3)
        ids = torch.tensor([[0, 1], [0, 1]])
        scores = torch.full((2, 2), 0.5)
        with mock.patch.object(runtime._MoEFunction, 'apply', return_value=torch.ones(4, 3)) as operation:
            result = runtime.MoonEPExecutor.__call__(executor, hidden, ids, scores, expert_weights=weights)
        torch.testing.assert_close(result, torch.ones_like(hidden), rtol=0, atol=0)
        operation.assert_called_once()
        self.assertEqual(completed, ['key'])
        self.assertNotIn('dedup_saved_rows', lifecycle.statistics)
