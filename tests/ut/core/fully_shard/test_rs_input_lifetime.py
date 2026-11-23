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
"""RS input retirement must preserve deferred output and synchronization ownership."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from hyper_parallel.core.fully_shard.hsdp_param import HSDPParamV2


class TestRSInputLifetime(unittest.TestCase):
    """Test input storage independently of sharded output and all-reduce handling."""

    def setUp(self) -> None:
        """Prepare independently owned RS input and output storage."""
        self.parameter = HSDPParamV2.__new__(HSDPParamV2)
        self.parameter._grad = torch.arange(16, dtype=torch.float32)
        self.parameter._grad_allocation_stream = None
        self.parameter._reduce_input_lease = None
        self.output = torch.arange(4, dtype=torch.float32)
        self.handle = Mock()
        self.parameter.reduce_scatter_comm_ctx = SimpleNamespace(
            reduce_scatter_handle=self.handle, reduce_scatter_output=self.output)

    def test_retirement_waits_before_free_and_preserves_output_handle(self):
        """A later consumer must still be able to wait on its own stream."""
        storage = self.parameter._grad.untyped_storage()
        self.handle.wait.side_effect = lambda: self.assertEqual(storage.nbytes(), 64)
        self.assertTrue(self.parameter.release_reduce_scatter_input())
        self.assertEqual(storage.nbytes(), 0)
        self.assertIsNone(self.parameter._grad)
        self.assertIs(self.parameter.reduce_scatter_comm_ctx.reduce_scatter_handle, self.handle)
        self.assertIs(self.parameter.reduce_scatter_comm_ctx.reduce_scatter_output, self.output)
        self.handle.wait.side_effect = None
        self.assertIs(self.parameter.reduce_scatter_output(), self.output)
        self.assertEqual(self.handle.wait.call_count, 2)
        torch.testing.assert_close(self.output, torch.arange(4, dtype=torch.float32), rtol=0, atol=0)
        self.assertIsNone(self.parameter.reduce_scatter_comm_ctx.reduce_scatter_handle)

    def test_repeated_retirement_does_not_free_or_wait_twice(self):
        """Iteration cleanup may revisit an input whose storage is already retired."""
        self.assertTrue(self.parameter.release_reduce_scatter_input())
        self.assertFalse(self.parameter.release_reduce_scatter_input())
        self.handle.wait.assert_called_once()

    def test_failed_wait_keeps_input_and_output_owned(self):
        """A communication error must not make live storage available for reuse."""
        original = self.parameter._grad
        self.handle.wait.side_effect = RuntimeError("collective failure")
        with self.assertRaisesRegex(RuntimeError, "collective failure"):
            self.parameter.release_reduce_scatter_input()
        self.assertIs(self.parameter._grad, original)
        self.assertEqual(original.untyped_storage().nbytes(), 64)
        self.assertIs(self.parameter.reduce_scatter_comm_ctx.reduce_scatter_output, self.output)

    def test_borrowed_input_returns_lease_without_resizing_pool_storage(self):
        """The pool owns physical allocation after the RS consumer has finished."""
        storage = self.parameter._grad.untyped_storage()
        lease = object()
        self.parameter._reduce_input_lease = lease
        self.parameter.reduce_input_pool = Mock()
        self.assertTrue(self.parameter.release_reduce_scatter_input())
        self.parameter.reduce_input_pool.release.assert_called_once_with(lease)
        self.assertEqual(storage.nbytes(), 64)
        self.assertIsNone(self.parameter._reduce_input_lease)

    def test_no_collective_keeps_local_gradient_unchanged(self):
        """Local-only reduction has no asynchronous input to retire."""
        self.parameter.reduce_scatter_comm_ctx.reduce_scatter_handle = None
        original = self.parameter._grad
        self.assertFalse(self.parameter.release_reduce_scatter_input())
        self.assertIs(self.parameter._grad, original)
