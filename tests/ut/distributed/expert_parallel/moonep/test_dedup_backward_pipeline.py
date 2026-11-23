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
"""CPU numerical and ordering contracts for the complete nonfused dedup backward."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep import planning, runtime, token_dedup
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestDedupBackwardPipeline(unittest.TestCase):
    """Compare real CPU expert gradients and two microsteps against ordinary autograd."""

    def test_reference_gradients_and_dispatch_precede_weight_gradients(self) -> None:
        """Keep empty experts, duplicate routes and prior weight gradients numerically correct."""
        torch.manual_seed(47)
        self.enterContext(patch.object(runtime, 'ep_all_to_all', side_effect=lambda value, *args: value.clone()))
        self.enterContext(patch.object(token_dedup, 'ep_all_to_all', side_effect=lambda value, *args: value.clone()))
        for top_k in (1, 2):
            for dtype in (torch.float32, torch.float64):
                for empty_expert in (False, True):
                    with self.subTest(top_k=top_k, dtype=dtype, empty_expert=empty_expert):
                        self._check_case(top_k, dtype, empty_expert)

    def test_forward_only_uses_the_original_backward_and_weight_accumulation(self) -> None:
        """Compact X while preserving source weighting and two microsteps of expert gradients."""
        torch.manual_seed(48)
        self.enterContext(patch.object(runtime, 'ep_all_to_all', side_effect=lambda value, *args: value.clone()))
        for top_k in (1, 2):
            for dtype in (torch.float32, torch.float64):
                for empty_expert in (False, True):
                    with self.subTest(top_k=top_k, dtype=dtype, empty_expert=empty_expert):
                        self._check_case(top_k, dtype, empty_expert, forward_only=True)

    def _check_case(self, top_k: int, dtype: torch.dtype, empty_expert: bool,
                    forward_only: bool = False) -> None:
        tokens, hidden, inner, experts = 5, 4, 3, 2
        ids = torch.tensor([[0, 0][:top_k] if empty_expert else [index % 2, (index + 1) % 2][:top_k]
                            for index in range(tokens)])
        counts = torch.bincount(ids.flatten(), minlength=experts)[None, :]
        plan = planning.compile_plan(counts, counts.sum(0)[:, None], ids.flatten(), 0, replica_capacity=0)
        if forward_only:
            dedup = token_dedup.prepare_forward_dedup(
                plan.dispatch_order, plan.token_send, plan.token_recv, top_k, None,
                forward_ratio=1.0, target_order=plan.sort_order)
        else:
            local = token_dedup.prepare_local_dedup(plan.dispatch_order, plan.token_send, plan.token_recv,
                                                   top_k, forward_ratio=1, backward_ratio=1)
            dedup = token_dedup.finish_dedup(local, None, device=torch.device('cpu'), width=hidden,
                                             source_inverse=plan.unpermute_indices, target_order=plan.sort_order)
        plan = replace(plan, token_transport=dedup)
        key = object()
        gate = (torch.randn(experts, hidden, 2 * inner, dtype=dtype) * .1).requires_grad_()
        down = (torch.randn(experts, inner, hidden, dtype=dtype) * .1).requires_grad_()
        reference_gate = gate.detach().clone().requires_grad_()
        reference_down = down.detach().clone().requires_grad_()
        executor = SimpleNamespace(owner=torch.nn.Module(), layer=0, top_k=top_k, ranks=1,
                                   group=None, gradient_group=None, limit=None, activation='swiglu')
        lifecycle = SimpleNamespace(executions={key: (executor, plan)}, fsdp_layers=set(),
                                    consume=lambda invocation: (gate, down), release=Mock(), prepare=Mock(),
                                    prefetch_previous=Mock(), complete=Mock(), prefetch_after_backward=Mock(),
                                    gradient_stream=Mock(return_value=None))
        for _ in range(2):
            value = torch.randn(1, tokens, hidden, dtype=dtype, requires_grad=True)
            scores = torch.randn(1, tokens, top_k, dtype=torch.float32, requires_grad=True)
            reference_value = value.detach().clone().requires_grad_()
            reference_scores = scores.detach().clone().requires_grad_()
            expected = []
            for token in range(tokens):
                pieces = []
                for slot in range(top_k):
                    expert = int(ids[token, slot])
                    preact = reference_value[0, token] @ reference_gate[expert]
                    first, second = preact.chunk(2)
                    output = (torch.nn.functional.silu(first) * second) @ reference_down[expert]
                    pieces.append(output * reference_scores[0, token, slot])
                expected.append(sum(pieces))
            expected = torch.stack(expected)
            if forward_only:
                source_rows = plan.dispatch_order // top_k
                returned = runtime._MoEFunction.apply(value, gate, down, key, lifecycle, source_rows)
                actual = runtime._combine_returned_tokens(value, returned, scores, plan, source_rows).reshape(tokens, hidden)
            else:
                actual = runtime._DedupMoEFunction.apply(value, scores, gate, down, key, lifecycle)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
            order = []
            pack = token_dedup.TokenDedup.pack_backward
            weight_grad = runtime.grouped_weight_grad
            restore_rows = runtime._restore_expert_rows

            def record_pack(instance, *args):
                order.append('BC_pack')
                return pack(instance, *args)

            def record_restore_rows(*args, **kwargs):
                order.append('BC_copy')
                return restore_rows(*args, **kwargs)

            def record_weight_grad(*args):
                order.append('weight_gradient')
                return weight_grad(*args)

            derivative = torch.randn_like(actual)
            with patch.object(token_dedup.TokenDedup, 'pack_backward', record_pack), \
                    patch.object(runtime, 'grouped_weight_grad', record_weight_grad), \
                    patch.object(runtime, '_restore_expert_rows', record_restore_rows):
                actual.backward(derivative)
            expected.backward(derivative)
            self.assertEqual(order, ['BC_copy' if forward_only else 'BC_pack', 'weight_gradient', 'weight_gradient'])
            for tensor, reference in zip((value, scores, gate, down),
                                         (reference_value, reference_scores, reference_gate, reference_down)):
                torch.testing.assert_close(tensor.grad, reference.grad, rtol=2e-5, atol=2e-7)
        self.assertEqual(lifecycle.complete.call_count, 2)
        self.assertEqual(lifecycle.release.call_count, 4)
