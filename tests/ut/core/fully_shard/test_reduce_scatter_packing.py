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
"""Numerical and layout acceptance for direct reduce-scatter packing."""
import unittest

import torch

from hyper_parallel.core.fully_shard.hsdp_param import pack_reduce_scatter_input


class TestReduceScatterPacking(unittest.TestCase):
    """Compare with the established cast-then-cat contract."""

    def test_shard_layouts_and_dtypes(self):
        """Destination rank and element order must be bitwise identical."""
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            original = torch.randn(4, 12, 8).to(dtype)
            for source in (original, original.transpose(1, 2)):
                for dim in range(3):
                    for reduction_dtype in (torch.float32, dtype):
                        with self.subTest(dtype=dtype, shape=source.shape, dim=dim, target=reduction_dtype):
                            reference = torch.cat(source.to(reduction_dtype).chunk(2, dim=dim), dim=0).flatten()
                            actual = pack_reduce_scatter_input(source, reduction_dtype, 2, dim)
                            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                            self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    def test_invalid_geometry(self):
        """Reject uneven shards before writing any communication input."""
        with self.assertRaises(ValueError):
            pack_reduce_scatter_input(torch.zeros(2, 7, 4), torch.float32, 2, 1)
