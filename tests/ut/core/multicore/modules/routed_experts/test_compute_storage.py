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
"""Byte-range admission for FP32 send writers beside live BF16 token input."""
import unittest

import torch

from hyper_parallel.core.multicore.modules.routed_experts.compute import _check_output_aliases


class TestMappedOutputStorage(unittest.TestCase):
    """Keep generic storage guards strict and mapped dense regions disjoint."""

    def test_mixed_dtype_adjacent_dense_ranges_are_safe_only_when_admitted(self):
        """Offsets must be compared in bytes across BF16/FP32 interpretations."""
        storage = torch.empty(128, dtype=torch.float32)
        derivative = storage.view(torch.bfloat16)[:64].view(8, 8)
        output = storage[32:64].view(8, 4)
        with self.assertRaisesRegex(ValueError, "alias"):
            _check_output_aliases(output, derivative)
        _check_output_aliases(output, derivative, allow_disjoint_storage=True)

    def test_one_float_overlap_is_rejected(self):
        """Even a one-element intrusion could corrupt a home dW input."""
        storage = torch.empty(128, dtype=torch.float32)
        derivative = storage.view(torch.bfloat16)[:64].view(8, 8)
        output = storage[31:63].view(8, 4)
        with self.assertRaisesRegex(ValueError, "alias"):
            _check_output_aliases(output, derivative, allow_disjoint_storage=True)

    def test_strided_storage_is_not_authorized_by_dense_byte_count(self):
        """The range exception only applies to tensors already validated as dense."""
        storage = torch.empty(128, dtype=torch.float32)
        derivative = storage[:64:2]
        output = storage[64:96]
        with self.assertRaisesRegex(ValueError, "alias"):
            _check_output_aliases(output, derivative, allow_disjoint_storage=True)
