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
"""Tensor grouped-matmul contracts, including empty expert groups."""
import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep.numerics import grouped_values
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestGroupedValues(unittest.TestCase):
    """Keep expert order and transpose behavior across zero-row groups."""

    def test_empty_group_and_transposed_weights(self):
        """A zero-token expert must not shift later experts in either direction."""
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn(3, 4, 6, generator=generator).to(torch.bfloat16)
        boundaries = torch.tensor([3, 3, 8])
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                matrices = weight.transpose(-1, -2) if transpose else weight
                x = torch.randn(8, matrices.shape[-2], generator=generator).to(torch.bfloat16)
                expected = torch.cat((x[:3] @ matrices[0], x[3:] @ matrices[2]))
                torch.testing.assert_close(grouped_values(x, matrices, boundaries), expected, rtol=0, atol=0)

    def test_entirely_empty_input(self):
        """Empty batches retain the output width and BF16 dtype."""
        weight = torch.empty(3, 4, 6, dtype=torch.bfloat16)
        result = grouped_values(torch.empty(0, 4, dtype=torch.bfloat16), weight, torch.zeros(3, dtype=torch.int64))
        self.assertEqual(result.shape, (0, 6))
        self.assertEqual(result.dtype, torch.bfloat16)
