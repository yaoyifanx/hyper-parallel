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
"""CPU checks for target-peer token deduplication metadata."""

import math
import unittest
from typing import Sequence
from unittest.mock import Mock, patch

import numpy as np
import torch

from hyper_parallel.distributed.expert_parallel.moonep import token_dedup
from hyper_parallel.distributed.expert_parallel.moonep import peer_token_plan
from hyper_parallel.distributed.expert_parallel.moonep.token_dedup import (
    failed_local_dedup, finish_dedup, prepare_local_dedup,
)


def _layout_reference(vector_counts: Sequence[int], score_counts: Sequence[int], width: int,
                    *, include_pack: bool = True) -> tuple:
    """Place peer vectors and scalar tails with no per-peer device operations."""
    if width < 1 or len(vector_counts) != len(score_counts):
        raise ValueError("dedup peer layout requires a positive width and matching peer counts")
    vectors = np.asarray(vector_counts, dtype=np.int64)
    scores = np.asarray(score_counts, dtype=np.int64)
    tails = (scores + width - 1) // width
    splits = vectors + tails
    vector_offsets = np.cumsum(vectors) - vectors
    score_offsets = np.cumsum(scores) - scores
    row_offsets = np.cumsum(splits) - splits
    tail_offsets = np.cumsum(tails) - tails
    total_vectors, total_scores = int(vectors.sum()), int(scores.sum())

    if not include_pack:
        vector_peers = np.repeat(np.arange(len(vectors)), vectors)
        vector_local = np.arange(total_vectors) - np.repeat(vector_offsets, vectors)
        vector_rows = row_offsets[vector_peers] + vector_local
        score_peers = np.repeat(np.arange(len(scores)), scores)
        score_local = np.arange(total_scores) - np.repeat(score_offsets, scores)
        score_elements = (row_offsets[score_peers] + vectors[score_peers]) * width + score_local
        return vector_rows, score_elements, tuple(splits.tolist()), None, None

    output_peers = np.repeat(np.arange(len(splits)), splits)
    output_local = np.arange(int(splits.sum())) - np.repeat(row_offsets, splits)
    vector_mask = output_local < vectors[output_peers]
    pack_order = np.where(vector_mask, vector_offsets[output_peers] + output_local,
                          total_vectors + tail_offsets[output_peers] + output_local - vectors[output_peers])

    padded = tails * width
    padded_peers = np.repeat(np.arange(len(padded)), padded)
    padded_local = np.arange(int(padded.sum())) - np.repeat(np.cumsum(padded) - padded, padded)
    valid = padded_local < scores[padded_peers]
    score_pad = np.full(len(padded_local), total_scores, dtype=np.int64)
    score_pad[valid] = score_offsets[padded_peers[valid]] + padded_local[valid]
    return None, None, tuple(splits.tolist()), pack_order, score_pad



class TestLocalTokenDedup(unittest.TestCase):
    """The bounded count table must preserve sorted unique and inverse semantics."""

    def test_direct_expert_rows_share_target_metadata_admission(self) -> None:
        """Native consumers receive final expert coordinates without a second remap."""
        order = torch.tensor([0, 2, 1, 3])
        local = prepare_local_dedup(order, (4,), (4,), 2, forward_ratio=1.0,
                                    capabilities=32, keep_groups=True)
        values = torch.tensor([3, 0, 2, 1], dtype=torch.int32)
        expected = peer_token_plan.target_tables(local.packet, local.counts[1], 4, 2, 32, row_values=values)
        prepare = Mock()
        result = finish_dedup(local, None, device=torch.device('cpu'), width=16,
                              target_row_values=values, prepare=prepare)
        prepare.assert_called_once_with()
        torch.testing.assert_close(result.forward_target_members.reshape(-1, 2), expected[4][:sum(result.forward_recv)])
        torch.testing.assert_close(result.backward_target_members.reshape(-1, 2), expected[5][:sum(result.backward_recv)])

    def test_independent_preparation_precedes_metadata_wait(self) -> None:
        """The received packet is unreadable until wait, while independent work may run first."""
        local = prepare_local_dedup(torch.arange(4), (2, 2), (2, 2), 2)
        calls = []

        def exchange(output, packet, **kwargs):
            self.assertTrue(kwargs['async_op'])
            calls.append('submit')

            def wait():
                self.assertEqual(calls, ['submit', 'prepare'])
                output.copy_(packet)
                calls.append('wait')

            return Mock(wait=wait)

        def prepare():
            calls.append('prepare')

        with patch.object(token_dedup.dist, 'all_to_all_single', side_effect=exchange), \
                patch.object(token_dedup.dist, 'all_reduce') as reduce:
            finish_dedup(local, None, device=torch.device('cpu'), width=16, prepare=prepare)
        self.assertEqual(calls, ['submit', 'prepare', 'wait'])
        reduce.assert_called_once()

    def test_preparation_failure_reaches_the_common_status_boundary(self) -> None:
        """Independent local failures must reject payload instead of bypassing admission."""
        local = prepare_local_dedup(torch.arange(4), (4,), (4,), 2)
        with self.assertRaisesRegex(RuntimeError, 'metadata or preparation failed'):
            finish_dedup(local, None, device=torch.device('cpu'), width=16,
                          prepare=Mock(side_effect=ValueError('independent preparation failed')))

    def _check_reference(self, ranks: int, top_k: int, tokens: int, reorder: bool) -> None:
        rng = np.random.default_rng(71)
        rows = tokens * top_k
        order = rng.permutation(rows).astype(np.int64)
        send = np.bincount(rng.integers(ranks, size=rows), minlength=ranks)
        source_rows = rng.permutation(rows).astype(np.int64) if reorder else None
        choices = order if source_rows is None else order[source_rows]
        peers = np.repeat(np.arange(ranks, dtype=np.int64), send)
        stride = rows + 1
        keys = peers * stride + choices // top_k
        unique, inverse = np.unique(keys, return_inverse=True)
        counts = np.bincount(unique // stride, minlength=ranks)
        offsets = np.cumsum(counts) - counts

        local = prepare_local_dedup(torch.from_numpy(choices), send, send, top_k,
                                    forward_ratio=1.0)

        self.assertEqual(tuple(local.forward_send.tolist()), tuple(counts.tolist()))
        np.testing.assert_array_equal(local.forward_tokens[:sum(counts)].numpy(), unique % stride)
        np.testing.assert_array_equal(local.source_choices.numpy(), choices)
        np.testing.assert_array_equal(local.packet.numpy()[3 * (np.arange(rows) + peers + 1)],
                                      inverse - offsets[peers])
        np.testing.assert_array_equal(local.packet.numpy()[3 * (np.cumsum(send + 1) - send - 1)], counts)

    def test_bounded_table_matches_sort_at_ep16_topk8(self) -> None:
        """Exercise the dense-key fast path with realistic route dimensions."""
        self._check_reference(16, 8, 4096, False)
        self._check_reference(16, 8, 128, True)

    def test_sparse_key_domain_and_empty_route_match_sort(self) -> None:
        """Large key domains use the sorting fallback; empty routes stay valid."""
        self._check_reference(16, 1, 32, False)
        self._check_reference(1, 1, 0, False)
        empty = prepare_local_dedup(torch.empty(0, dtype=torch.int64), (0,), (0,), 1)
        self.assertEqual(finish_dedup(empty, None, device=torch.device("cpu"), width=16).saved_rows, 0)


    def test_mixed_peer_modes_share_one_metadata_packet(self) -> None:
        """The peer threshold preserves compact and dense source rows in one route."""
        order = torch.tensor([0, 1, 2, 4, 3, 5, 6, 7])
        local = prepare_local_dedup(order, (2, 2, 4), (2, 2, 4),
                                    2, forward_ratio=0.5, backward_ratio=1.0)
        self.assertEqual(tuple(local.forward_send.tolist()), (1, 2, 4))
        self.assertEqual(tuple(local.backward_send.tolist()), (1, 2, 3))
        np.testing.assert_array_equal(local.forward_tokens[:7].numpy(), [0, 1, 2, 1, 2, 3, 3])
        np.testing.assert_array_equal(local.backward_tokens[:6].numpy(), [0, 1, 2, 1, 2, 3])
        np.testing.assert_array_equal(local.packet.numpy()[[0, 9, 18]], [1, 2, 3])
        np.testing.assert_array_equal(local.packet.numpy()[[3, 6, 12, 15, 21, 24, 27, 30]],
                                      [0, 0, 0, 1, 0, 1, 2, 2])
        forward_dense = prepare_local_dedup(order, (2, 2, 4), (2, 2, 4),
                                            2, forward_ratio=0.0, backward_ratio=1.0)
        backward_dense = prepare_local_dedup(order, (2, 2, 4), (2, 2, 4),
                                             2, forward_ratio=1.0, backward_ratio=0.0)
        self.assertEqual(tuple(forward_dense.forward_send.tolist()), (2, 2, 4))
        self.assertEqual(tuple(forward_dense.backward_send.tolist()), (1, 2, 3))
        self.assertEqual(tuple(backward_dense.forward_send.tolist()), (1, 2, 3))
        self.assertEqual(tuple(backward_dense.backward_send.tolist()), (2, 2, 4))
        def exchange(output, payload, **_kwargs):
            output.copy_(payload)
            return Mock()

        with patch.object(token_dedup.dist, "all_to_all_single", side_effect=exchange), \
                patch.object(token_dedup.dist, "all_reduce"):
            for selected, saved in ((local, 1), (forward_dense, 0), (backward_dense, 2)):
                plan = finish_dedup(selected, None, device=torch.device("cpu"), width=16)
                self.assertEqual(plan.saved_rows, saved)

    def test_failed_local_packet_reaches_every_peer(self) -> None:
        """An invalid header travels through the normal fixed-size metadata exchange."""
        local = failed_local_dedup((2, 0, 3), (1, 2, 2))
        self.assertEqual(local.packet_send, (9, 3, 12))
        np.testing.assert_array_equal(local.packet.numpy()[[0, 9, 12]], [-1, -1, -1])

    def test_fp32_scores_preserve_bits_and_gradients_in_bf16_payload(self) -> None:
        """Feature: FP32 router-score transport with BF16 activations.

        Description: Carry each score through two BF16 payload lanes and differentiate the result.
        Expectation: Both received score bits and score gradients exactly match the source values.
        """
        order = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
        for width in (16, 17):
            with self.subTest(width=width):
                local = prepare_local_dedup(order, (4,), (4,), 2, forward_ratio=1.0)
                plan = finish_dedup(local, None, device=torch.device("cpu"), width=width, score_lanes=2)
                hidden = torch.randn(2, width, dtype=torch.bfloat16, requires_grad=True)
                scores = torch.tensor([[0.1234567, 0.8765433], [0.3333333, 0.6666667]],
                                      dtype=torch.float32, requires_grad=True)
                with patch.object(token_dedup, "ep_all_to_all", side_effect=lambda payload, *_args: payload.clone()):
                    received, received_scores = plan.dispatch_with_scores(hidden, scores)
                    (received.float() * received_scores[:, None]).sum().backward()
                torch.testing.assert_close(received_scores, scores.flatten().index_select(0, order), rtol=0, atol=0)
                expected = hidden.detach().float().sum(-1, keepdim=True).expand_as(scores)
                torch.testing.assert_close(scores.grad, expected, rtol=0, atol=0)

    def test_four_direction_gradients_match_fp64_expert_oracle(self) -> None:
        """Expert-major receive maps retain independent dProb and dW, including zero scores."""
        generator = torch.Generator().manual_seed(118)
        expert_ids = torch.tensor([[2, 0, 1], [1, 2, 0], [0, 1, 2]])
        order = expert_ids.flatten().argsort(stable=True)
        local = prepare_local_dedup(order, (9,), (9,), 3, forward_ratio=1.0)
        target_order = torch.arange(8, -1, -1)
        plan = finish_dedup(local, None, device=torch.device("cpu"), width=4, target_order=target_order)
        originals = (
            torch.randn(3, 4, generator=generator) * 0.2,
            torch.tensor([[0.0, 0.13, 0.87], [1.e-7, 0.31, 0.6899999], [0.2, 0.3, 0.5]]),
            torch.randn(3, 4, 5, generator=generator) * 0.2,
            torch.randn(3, 5, 4, generator=generator) * 0.2,
        )
        actual = tuple(value.clone().requires_grad_() for value in originals)
        reference = tuple(value.double().requires_grad_() for value in originals)
        dy = torch.randn(3, 4, generator=generator)
        hidden, scores, first, second = actual
        routed_experts = expert_ids.flatten().index_select(0, order).index_select(0, target_order)
        with patch.object(token_dedup, "ep_all_to_all", side_effect=lambda value, *_args: value.clone()):
            received, received_scores = plan.dispatch_with_scores(hidden, scores)
            intermediate = torch.bmm(received[:, None], first[routed_experts]).tanh()
            values = torch.bmm(intermediate, second[routed_experts]).squeeze(1)
            result = plan.return_vectors(values * received_scores[:, None], 3)
            gradients = torch.autograd.grad(result, actual, dy)
        ref_hidden, ref_scores, ref_first, ref_second = reference
        ref_intermediate = torch.einsum("th,tkhi->tki", ref_hidden, ref_first[expert_ids]).tanh()
        ref_values = torch.einsum("tki,tkih->tkh", ref_intermediate, ref_second[expert_ids])
        expected = (ref_values * ref_scores[:, :, None]).sum(1)
        expected_gradients = torch.autograd.grad(expected, reference, dy.double())
        torch.testing.assert_close(result.double(), expected, atol=1.e-5, rtol=1.e-5)
        for observed, oracle in zip(gradients, expected_gradients):
            torch.testing.assert_close(observed.double(), oracle, atol=1.e-5, rtol=1.e-5)
        self.assertNotEqual(gradients[1][0, 0].item(), 0.0)

    def test_fixed_capacity_device_tables_match_existing_peer_layouts(self) -> None:
        """Device-style grouping retains wire order, mixed modes and zero splits."""
        rng = np.random.default_rng(118)
        for ranks, top_k, tokens in ((3, 6, 17), (16, 8, 128), (16, 1, 32), (1, 1, 0)):
            rows = tokens * top_k
            order = torch.from_numpy(rng.permutation(rows))
            counts = np.bincount(rng.integers(ranks, size=rows), minlength=ranks)
            for forward, backward in ((0.75, 1.0), (0.0, 0.0), (1.0, 1.0)):
                with self.subTest(ranks=ranks, top_k=top_k, tokens=tokens, ratios=(forward, backward)):
                    controls = torch.tensor([
                        counts.tolist(), [math.floor(forward * int(count)) for count in counts],
                        [math.ceil(backward * int(count)) - 1 for count in counts]], dtype=torch.int32)
                    source = peer_token_plan.source_tables(order, controls, top_k)
                    minimal = peer_token_plan.source_tables(order, controls, top_k, keep_groups=False)
                    self.assertEqual(len(minimal), 8)
                    for observed, expected in zip(minimal[:7], source[:7]):
                        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
                    torch.testing.assert_close(minimal[-1], source[-1], atol=0, rtol=0)
                    target = peer_token_plan.target_tables(source[0], controls[0], rows, top_k)
                    choices = order.numpy()
                    peers = np.repeat(np.arange(ranks), counts)
                    keys = peers * (tokens + 1) + choices // top_k
                    unique, inverse = np.unique(keys, return_inverse=True)
                    raw_counts = np.bincount(unique // (tokens + 1), minlength=ranks)
                    forward_counts = np.where(raw_counts <= forward * counts, raw_counts, counts)
                    backward_counts = np.where(raw_counts < backward * counts, raw_counts, counts)
                    raw_slot = inverse - (np.cumsum(raw_counts) - raw_counts)[peers]
                    dense_slot = np.arange(rows) - (np.cumsum(counts) - counts)[peers]
                    self.assertTrue(bool(source[-1]) and bool(target[-1]))
                    np.testing.assert_array_equal(source[3].numpy(), forward_counts)
                    np.testing.assert_array_equal(source[4].numpy(), backward_counts)
                    for actual, compact, selected in (
                            (target[2], raw_counts <= forward * counts, forward_counts),
                            (target[3], raw_counts < backward * counts, backward_counts)):
                        expected = (np.cumsum(selected) - selected)[peers] + np.where(
                            compact[peers], raw_slot, dense_slot)
                        np.testing.assert_array_equal(actual.numpy(), expected)
                    for members, lookup, capacity in ((target[4], target[2], int(source[3].sum())),
                                                       (target[5], target[3], int(source[4].sum()))):
                        for group in range(capacity):
                            expected = torch.where(lookup == group)[0].int()
                            observed = members[group][members[group] >= 0]
                            torch.testing.assert_close(observed, expected)
                    for width in (16, 5120):
                        layout = peer_token_plan.layout_indices(source[3], controls[0] * 2, width, rows, rows * 2)
                        vector_rows, score_elements, splits, _, _ = _layout_reference(
                            source[3].tolist(), (controls[0] * 2).tolist(), width, include_pack=False)
                        _, _, _, pack_order, _ = _layout_reference(
                            source[3].tolist(), (controls[0] * 2).tolist(), width)
                        scores = counts * 2
                        score_peers = np.repeat(np.arange(ranks), scores)
                        tails = (scores + width - 1) // width
                        expected_positions = (np.cumsum(tails) - tails)[score_peers] * width + np.arange(
                            rows * 2) - (np.cumsum(scores) - scores)[score_peers]
                        np.testing.assert_array_equal(layout[0][:int(source[3].sum())].numpy(), vector_rows)
                        np.testing.assert_array_equal(layout[1].numpy(), score_elements)
                        np.testing.assert_array_equal(layout[2][:sum(splits)].numpy(), pack_order)
                        np.testing.assert_array_equal(layout[3].numpy(), expected_positions)

    def test_invalid_device_headers_are_rejected_without_out_of_bounds(self) -> None:
        """Malformed control data produces poison status before any payload phase."""
        counts = torch.tensor([4], dtype=torch.int32)
        controls = torch.stack((counts, counts, counts - 1))
        packet = peer_token_plan.source_tables(torch.arange(4), controls, 2)[0]
        for index, invalid in ((0, 100000), (0, -1), (2, 1), (3, 100000), (5, 100000)):
            with self.subTest(index=index, value=invalid):
                malformed = packet.clone()
                malformed[index] = invalid
                result = peer_token_plan.target_tables(malformed, counts, 4, 2)
                self.assertFalse(bool(result[-1]))

    def test_packed_plan_outputs_keep_independent_execution_storage(self) -> None:
        """One backing allocation per dtype must not let the next call overwrite a saved plan."""
        order = torch.arange(12)
        controls = torch.tensor([[6, 6], [6, 6], [5, 5]], dtype=torch.int32)
        def program(choices: torch.Tensor, counts: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return peer_token_plan.source_tables(choices, counts, 3)

        first = token_dedup.materialize_owned_tables(program, order, controls)
        saved = tuple(value.clone() for value in first)
        second = token_dedup.materialize_owned_tables(program, order.flip(0), controls)
        for observed, expected in zip(first, saved):
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        for outputs in (first, second):
            storages = {value.untyped_storage().data_ptr() for value in outputs}
            self.assertEqual(len(storages), len({value.dtype for value in outputs}))
        self.assertNotEqual(first[0].untyped_storage().data_ptr(), second[0].untyped_storage().data_ptr())

    def test_materialize_owned_tables_own_aliases_across_dynamic_shapes(self) -> None:
        """Saved views survive input reuse and later invocations with different extents."""
        def program(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return values, values[::2], values.sum(), values > 0, values[:0]

        first_input = torch.arange(12, dtype=torch.int32)
        expected = tuple(value.clone() for value in program(first_input))
        first = token_dedup.materialize_owned_tables(program, first_input)
        second_input = torch.arange(7, dtype=torch.int32)
        second = token_dedup.materialize_owned_tables(program, second_input)
        first_input.fill_(-1)
        for observed, saved in zip(first, expected):
            torch.testing.assert_close(observed, saved, atol=0, rtol=0)
        for observed, saved in zip(second, program(second_input)):
            torch.testing.assert_close(observed, saved, atol=0, rtol=0)
        self.assertNotEqual(first[0].untyped_storage().data_ptr(), first_input.untyped_storage().data_ptr())
        self.assertNotEqual(first[0].untyped_storage().data_ptr(), second[0].untyped_storage().data_ptr())
