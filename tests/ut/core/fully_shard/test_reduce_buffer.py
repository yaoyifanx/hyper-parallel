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
"""RS lease budget and in-flight storage isolation."""
import unittest

import torch

from hyper_parallel.core.fully_shard.reduce_buffer import ReduceBufferPool


class TestReduceBufferPool(unittest.TestCase):
    """An unfinished collective must retain exclusive ownership of its input."""

    def test_budget_and_exclusive_lease(self):
        """A full pool falls back without overwriting a busy input."""
        pool = ReduceBufferPool(128)
        first, token = pool.acquire((32,), torch.float32, torch.device("cpu"))
        first.fill_(17)
        second, second_token = pool.acquire((32,), torch.float32, torch.device("cpu"))
        self.assertIsNone(second_token)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        second.zero_()
        torch.testing.assert_close(first, torch.full_like(first, 17))
        pool.release(token)
        third, third_token = pool.acquire((16,), torch.float32, torch.device("cpu"))
        self.assertEqual(first.data_ptr(), third.data_ptr())
        self.assertEqual(pool.reserved_bytes, 128)
        pool.release(third_token)
        with self.assertRaises(ValueError):
            pool.release(third_token)

    def test_evict_idle_storage(self):
        """A new dtype/shape may replace idle slots within the same byte budget."""
        pool = ReduceBufferPool(128)
        first, token = pool.acquire((16,), torch.float32, torch.device("cpu"))
        pool.release(token)
        second, token = pool.acquire((64,), torch.bfloat16, torch.device("cpu"))
        self.assertEqual(second.dtype, torch.bfloat16)
        self.assertLessEqual(pool.reserved_bytes, 128)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        pool.release(token)
