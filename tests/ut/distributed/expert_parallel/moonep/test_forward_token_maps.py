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
"""CPU protocol oracle for compact forward X and unchanged selection returns."""

import math
import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep import peer_token_plan
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestForwardTokenMaps(unittest.TestCase):
    """Exchange real descriptor slices and compare original source token identities."""

    def test_variable_sources_thresholds_and_expert_coordinates(self) -> None:
        """Keep empty peers and unequal source extents correct in both compact and dense modes."""
        torch.manual_seed(51)
        for ranks, top_k, tokens in ((1, 1, 0), (1, 2, 5), (2, 3, 9), (4, 8, 13)):
            for ratio in (0.0, .75, 1.0):
                with self.subTest(ranks=ranks, top_k=top_k, tokens=tokens, ratio=ratio):
                    self._check_topology(ranks, top_k, tokens, ratio)

    def _check_topology(self, ranks: int, top_k: int, tokens: int, ratio: float) -> None:
        packets, sources, base_rows, source_counts = [], [], [], []
        for rank in range(ranks):
            rows = (tokens + rank % 2 if tokens else 0) * top_k
            permutation = torch.randperm(rows)
            destinations = torch.randint(ranks, (rows,))
            order = permutation.index_select(0, torch.argsort(destinations, stable=True))
            counts = torch.bincount(destinations, minlength=ranks).int()
            controls = torch.stack((counts, counts, counts.new_tensor(
                [math.floor(int(count) * ratio) for count in counts])))
            packet, forward, slots, ids = peer_token_plan.source_forward_tables(order, controls, top_k)
            actual_tokens = peer_token_plan._scatter_rows(torch.zeros(rows, dtype=torch.int32), slots, ids)
            reference, expected_counts = [], []
            offset = 0
            for count in counts.tolist():
                selected = (order[offset:offset + count] // top_k).tolist()
                unique = sorted(set(selected))
                output = unique if len(unique) <= math.floor(count * ratio) else selected
                reference.extend(output)
                expected_counts.append(len(output))
                offset += count
            self.assertEqual(forward.tolist(), expected_counts)
            self.assertEqual(actual_tokens[:sum(expected_counts)].tolist(), reference)
            packets.append(torch.split(packet, tuple((counts + 1).tolist())))
            sources.append((actual_tokens, forward))
            base_rows.append(order // top_k + rank * 10000)
            source_counts.append(counts)
        for target in range(ranks):
            counts = torch.tensor([int(source_counts[source][target]) for source in range(ranks)], dtype=torch.int32)
            rows = int(counts.sum())
            order = torch.randperm(rows)
            packet = torch.cat([packets[source][target] for source in range(ranks)])
            _, lookup, valid = peer_token_plan.target_forward_tables(packet, counts, rows, order)
            self.assertTrue(bool(valid))
            unique, expected = [], []
            for source in range(ranks):
                values, forward = sources[source]
                start = int(forward[:target].sum())
                count = int(forward[target])
                unique.append(values[start:start + count] + source * 10000)
                begin = int(source_counts[source][:target].sum())
                count = int(source_counts[source][target])
                expected.append(base_rows[source][begin:begin + count])
            actual = torch.cat(unique).index_select(0, lookup)
            reference = torch.cat(expected).index_select(0, order)
            torch.testing.assert_close(actual.long(), reference, rtol=0, atol=0)

    def test_invalid_receive_coordinates_are_rejected_without_out_of_bounds_reads(self) -> None:
        """Malformed headers and row slots reach the common device validity boundary."""
        counts = torch.tensor([4], dtype=torch.int32)
        controls = torch.stack((counts, counts, counts))
        packet = peer_token_plan.source_forward_tables(torch.arange(4), controls, 2)[0]
        for offset, value in ((0, -1), (0, 5), (1, -1), (1, 8)):
            broken = packet.clone()
            broken[offset] = value
            _, _, valid = peer_token_plan.target_forward_tables(broken, counts, 4, torch.arange(4))
            self.assertFalse(bool(valid))
