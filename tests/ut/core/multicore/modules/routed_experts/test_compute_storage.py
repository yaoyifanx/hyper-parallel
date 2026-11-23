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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_parallel.core.multicore.modules.routed_experts.compute import _check_output_aliases
from hyper_parallel.core.multicore.modules.routed_experts import compute


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


class TestConsumedCombineStorage(unittest.TestCase):
    """Allow only the explicitly consumed private value buffer to become dV."""

    def setUp(self):
        self.dy = torch.empty((2, 16), dtype=torch.bfloat16)
        self.values = torch.empty((4, 16), dtype=torch.bfloat16)
        self.order = torch.arange(4)
        self.scores = torch.empty((2, 2), dtype=torch.float32)
        self.native = SimpleNamespace(routed_experts_combine_backward=Mock(),
                                      routed_experts_combine_backward_consume=Mock())
        self.enterContext(patch.object(compute, "_load_native"))
        self.enterContext(patch.object(compute.torch.ops, "hyper_parallel", self.native))

    def test_explicit_consumption_preserves_the_exact_storage(self):
        """The specialized native entry owns both the final read and the dV store."""
        dv, ds = compute.combine_gradient(self.dy, self.values, self.order, self.scores, consume_values=True)
        self.assertIs(dv, self.values)
        self.assertEqual(ds.dtype, self.scores.dtype)
        self.native.routed_experts_combine_backward_consume.assert_called_once()
        self.native.routed_experts_combine_backward.assert_not_called()

    def test_default_contract_still_rejects_consumption(self):
        """An ordinary caller cannot silently overwrite its saved expert output."""
        with self.assertRaisesRegex(ValueError, "alias"):
            compute.combine_gradient(self.dy, self.values, self.order, self.scores,
                                     outputs=(self.values, torch.empty_like(self.scores)))

    def test_consumption_rejects_an_independent_output(self):
        """A same-shaped tensor cannot satisfy the exact ownership handoff."""
        with self.assertRaisesRegex(ValueError, "exact dV"):
            compute.combine_gradient(self.dy, self.values, self.order, self.scores,
                                     outputs=(self.values.clone(), torch.empty_like(self.scores)),
                                     consume_values=True)

    def test_consumption_keeps_other_readers_disjoint(self):
        """An order table sharing value storage must fail before kernel submission."""
        order = self.values.view(torch.int64).reshape(-1)[:4]
        with self.assertRaisesRegex(ValueError, "alias"):
            compute.combine_gradient(self.dy, self.values, order, self.scores, consume_values=True)
        self.native.routed_experts_combine_backward_consume.assert_not_called()

    def test_preactivation_consumption_requires_its_exact_output(self):
        """The derivative may overwrite private Z only through the consuming entry."""
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        engine.swiglu_limit = None
        self.native.routed_experts_swiglu_backward_consume = Mock()
        preact = torch.empty((2, 32), dtype=torch.bfloat16)
        gradient = torch.empty((2, 16), dtype=torch.bfloat16)
        engine._activation_gradient(gradient, preact, preact, consume_preactivation=True)
        self.native.routed_experts_swiglu_backward_consume.assert_called_once_with(gradient, preact)
        with self.assertRaisesRegex(ValueError, "exact derivative"):
            engine._activation_gradient(gradient, preact, preact.clone(), consume_preactivation=True)

    def test_clipped_activation_keeps_its_original_storage_contract(self):
        """Clipped producers retain separate Z/dZ until explicitly supported."""
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        engine.swiglu_limit = 7.0
        with self.assertRaisesRegex(ValueError, "unclipped"):
            engine._activation_gradient(self.dy, self.values, self.values, consume_preactivation=True)


class TestW2FirstOrdering(unittest.TestCase):
    """Require the complete W2 pass before dZ and the consuming dX tail."""

    def test_mapped_w2_pass_precedes_consuming_tail(self):
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        order = []
        work = SimpleNamespace(pipeline=True, da_work=object(), state=SimpleNamespace(preact=object()),
                               activation=object(), down_work=SimpleNamespace(output_table=object()),
                               early_experts=object(), da=object(), dz=object(), w2_before_input_grad=True,
                               consume_preactivation=True, down_gradient=object(), gate_gradient=object(),
                               output=object(), ready=object())
        with patch.object(engine, "_claim_unsubmitted_work"), \
             patch.object(engine, "run_linear", side_effect=lambda *args: order.append("dA")), \
             patch.object(engine, "_activate", side_effect=lambda *args: order.append("A")), \
             patch.object(engine, "run_mapped_gradient", side_effect=lambda *args: order.append("W2")), \
             patch.object(engine, "_activation_gradient", side_effect=lambda *args, **kwargs: order.append("dZ")), \
             patch.object(engine, "_run_backward_tail", side_effect=lambda *args: order.append("tail")):
            engine.run_backward(work, lambda role, value: order.append(role),
                                submit_return=lambda: order.append("return"))
        self.assertEqual(order, ["dA", "A", "W2", "dZ", "tail", "down", "gate", "return"])

    def test_consumption_requires_the_w2_completion_contract(self):
        engine = compute.NativeExpertCompute.__new__(compute.NativeExpertCompute)
        derivative = torch.empty((3, 16), dtype=torch.bfloat16)
        state = SimpleNamespace(x=derivative.clone(), preact=torch.empty((3, 32), dtype=torch.bfloat16))
        with patch.object(compute, "_load_native"), patch.object(engine, "_check_input"):
            with self.assertRaisesRegex(ValueError, "every W2"):
                engine.prepare_backward(derivative, state, [], [], pipeline=True, consume_derivative=True)
            with self.assertRaisesRegex(ValueError, "requires the backward pipeline"):
                engine.prepare_backward(derivative, state, [], [], w2_before_input_grad=True)
