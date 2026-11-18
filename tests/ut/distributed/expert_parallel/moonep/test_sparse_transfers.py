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
"""Bounded replica traffic and numerical oracle tests without distributed setup."""

import unittest

import torch

from tests.common.moonep.reference import (
    backward_values, combine, combine_gradient, forward_values, grouped_gradient, swiglu,
)
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import (
    compile_replica_transfer, copy_matrix,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestReplicaTransfer(unittest.TestCase):
    """Check fanout, split matrices, empty participants and inverse attribution."""

    def test_owner_fanout_uses_one_complete_exchange(self):
        """The real fanout exceeds home capacity while every peer has matching splits."""
        allocation = torch.zeros(128, 16, dtype=torch.int64)
        allocation[:8] = 1
        for reverse in (False, True):
            item = compile_replica_transfer(allocation, (("gate", 7), ("down", 5)), reverse=reverse)
            self.assertIsNotNone(item)
            self.assertEqual(len(item.fragments), 240)
            for rank in range(16):
                local = item.local(rank)
                sent, received = local.sent, local.received
                send_sizes, receive_sizes = local.send_sizes, local.receive_sizes
                self.assertEqual(sum(fragment.elements for fragment in sent), sum(send_sizes))
                self.assertEqual(sum(fragment.elements for fragment in received), sum(receive_sizes))
                self.assertIs(item.local(rank), item.local(rank))
                for peer in range(16):
                    self.assertEqual(send_sizes[peer], item.local(peer).receive_sizes[rank])
            for fragment in item.fragments:
                self.assertEqual(fragment.target if reverse else fragment.source, 0)
                self.assertEqual(fragment.elements, 7 if fragment.role == "gate" else 5)
            owner = item.local(0)
            self.assertEqual(sum(owner.receive_sizes if reverse else owner.send_sizes), 120 * 12)
        self.assertIsNone(compile_replica_transfer(torch.eye(2, dtype=torch.int64), (("gate", 7),)))

    def test_complete_copy_handles_transposed_borrowed_weights(self):
        """Full logical matrices pack and unpack without intermediate transposes."""
        matrix = torch.arange(35).reshape(5, 7).t()
        flat = torch.empty(matrix.numel(), dtype=matrix.dtype)
        copy_matrix(flat, matrix)
        torch.testing.assert_close(flat, matrix.reshape(-1), rtol=0, atol=0)
        target = torch.zeros_like(matrix)
        copy_matrix(flat, target, unpack=True)
        torch.testing.assert_close(target, matrix, rtol=0, atol=0)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestReferenceMath(unittest.TestCase):
    """Check forward, individual dW readiness and router derivatives against autograd."""

    def test_split_experts_and_gradients(self):
        """Independent matrix references give the same derivative as a dense loop."""
        generator = torch.Generator().manual_seed(233)
        x = torch.randn(9, 4, generator=generator, requires_grad=True)
        gate_up = [torch.randn(4, 6, generator=generator, requires_grad=True) for _ in range(3)]
        down = [torch.randn(3, 4, generator=generator, requires_grad=True) for _ in range(3)]
        boundaries = torch.tensor([2, 3, 9])
        output, state = forward_values(x, gate_up, down, boundaries, 3)
        parts, start = [], 0
        for index, end in enumerate(boundaries.tolist()):
            parts.append(swiglu(x[start:end] @ gate_up[index]) @ down[index])
            start = end
        expected = torch.cat(parts)
        dy = torch.randn(output.shape, generator=generator)
        expected.backward(dy)
        ready = []
        dx, dg, dd = backward_values(dy, state, gate_up, down, lambda role, value: ready.append(role))
        self.assertEqual(ready, ["down", "gate"])
        for actual, reference in ((output, expected), (dx, x.grad),
                                  (dg, torch.stack([weight.grad for weight in gate_up])),
                                  (dd, torch.stack([weight.grad for weight in down]))):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)

    def test_combine_derivatives(self):
        """Products follow ordinary EP's dtype promotion and rounding boundary."""
        for score_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(score_dtype=score_dtype):
                generator = torch.Generator().manual_seed(59)
                values = torch.randn(24, 16, generator=generator, dtype=torch.bfloat16, requires_grad=True)
                scores = torch.randn(6, 4, generator=generator, dtype=score_dtype, requires_grad=True)
                order = torch.randperm(24, generator=generator)
                rows = order // scores.shape[1]
                weighted = values * scores.flatten()[order].unsqueeze(-1)
                expected = weighted.new_zeros((6, 16)).index_add_(0, rows, weighted)
                dy = torch.randn(expected.shape, generator=generator, dtype=expected.dtype)
                expected.backward(dy)

                output = combine(values.detach(), order, scores.detach(), 6)
                dv, ds = combine_gradient(dy, values.detach(), order, scores.detach())

                torch.testing.assert_close(output, expected.detach(), rtol=0, atol=0)
                torch.testing.assert_close(dv, values.grad, rtol=0, atol=0)
                torch.testing.assert_close(ds, scores.grad, rtol=0, atol=0)
                if score_dtype == torch.bfloat16:
                    unrounded_ds = torch.empty_like(scores).flatten()
                    unrounded_ds[order] = (dy[rows].float() * values.detach().float()).sum(-1).to(score_dtype)
                    self.assertFalse(torch.equal(ds, unrounded_ds.view_as(scores)),
                                     f"rounded dscore={ds}, unrounded FP32 dot={unrounded_ds.view_as(scores)}")


    def test_bfloat16_partials_round_only_after_owner_merge(self):
        """A representable final sum must retain a partial at the BF16 midpoint."""
        x = torch.ones((3, 1), dtype=torch.bfloat16)
        dy = torch.tensor([[1.0], [1 / 256], [1 / 256]], dtype=torch.bfloat16)

        partials = grouped_gradient(x, dy, torch.tensor([2, 3]))
        complete = grouped_gradient(x, dy, torch.tensor([3]))[0]
        merged = partials.sum(0)
        early_rounded = partials.to(torch.bfloat16).sum(0)

        torch.testing.assert_close(partials[0], torch.tensor([[1 + 1 / 256]]), rtol=0, atol=0)
        torch.testing.assert_close(merged.double(), x.double().t() @ dy.double(), rtol=0, atol=0)
        torch.testing.assert_close(merged.to(torch.bfloat16), complete.to(torch.bfloat16), rtol=0, atol=0)
        self.assertFalse(torch.equal(early_rounded, complete.to(torch.bfloat16)),
                         f"early BF16 merge={early_rounded}, once-rounded result={complete.to(torch.bfloat16)}")
