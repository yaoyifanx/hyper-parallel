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
"""Chunk conservation, explicit gradients, replay and communication ordering on CPU."""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional
from torch.utils.checkpoint import DefaultDeviceType

from hyper_parallel.distributed.expert_parallel.moonep import configure_moonep, wrap_moonep_checkpoint
from hyper_parallel.distributed.expert_parallel.moonep import chunked, runtime as runtime_module
from hyper_parallel.distributed.expert_parallel.moonep import planning
from hyper_parallel.distributed.expert_parallel.moonep.chunked_math import store_rows
from hyper_parallel.distributed.expert_parallel.moonep.chunked_plan import compile_chunks
from hyper_parallel.distributed.expert_parallel.moonep.device_planning import (
    HostControl, inverse_permutation, peer_round_quotas,
)
from hyper_parallel.distributed.expert_parallel.moonep.layout import RoundGeometry, round_geometry, route_geometry
from hyper_parallel.distributed.expert_parallel.moonep.planning import compile_plan, select_allocation
from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPExecutor, local_home_gradients
from tests.common.mark_utils import arg_mark


class _Experts(torch.nn.Module):
    def __init__(self, activation: str = 'swiglu', transposed: bool = False) -> None:
        """Create a small expert owner supporting either physical weight layout."""
        super().__init__()
        self.transposed = transposed
        gate_up, down = torch.randn(3, 4, 6) * .2, torch.randn(3, 3, 4) * .2
        self.gate_up = torch.nn.Parameter(gate_up.transpose(1, 2).contiguous() if transposed else gate_up)
        self.down = torch.nn.Parameter(down.transpose(1, 2).contiguous() if transposed else down)
        self._moonep_executor = MoonEPExecutor(self, self.weights, ep_group=None, ep_size=1,
                                              num_experts=3, top_k=2, activation=activation, swiglu_limit=.25)

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logical views while retaining the real Parameter edges."""
        return ((self.gate_up.transpose(1, 2), self.down.transpose(1, 2)) if self.transposed
                else (self.gate_up, self.down))

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Execute the same owner boundary used by model adapters."""
        return self._moonep_executor(hidden, ids, scores, expert_weights=self.weights())


def _reference(model, hidden, ids, scores):
    gate_up, down = model.weights()
    result = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            left, right = (hidden[token] @ gate_up[expert]).chunk(2)
            if model._moonep_executor.activation == 'clipped':
                left, right = left.clamp(max=.25), right.clamp(-.25, .25)
            result[token] = result[token] + scores[token, choice] * ((functional.silu(left) * right) @ down[expert])
    return result


class TestChunkedMoonEP(unittest.TestCase):
    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_inverse_permutation_preserves_dtype_and_exact_indices(self):
        """Feature: Native and chunk inverse maps.
        Description: Exercise empty, singleton and nontrivial bijections in both integer dtypes.
        Expectation: Inversion preserves the consumer dtype and restores every original position.
        """
        for dtype in (torch.int32, torch.int64):
            for size in (0, 1, 17):
                with self.subTest(dtype=dtype, size=size):
                    order = torch.randperm(size).to(dtype)
                    inverse = inverse_permutation(order)
                    self.assertEqual(inverse.dtype, dtype)
                    torch.testing.assert_close(inverse.index_select(0, order.long()),
                                               torch.arange(size, dtype=dtype))

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_chunks_match_full_placement_and_reciprocal_splits(self):
        """Feature: Receiver-bounded chunk planning.
        Description: Simulate every peer exchange for random, skewed and empty sources.
        Expectation: All choices occur once and each received chunk equals its full-plan slice.
        """
        torch.manual_seed(41)
        for ranks in (1, 2, 4, 16):
            counts = torch.randint(0, 5, (ranks, ranks * 2), dtype=torch.int64)
            counts[0] = 0
            if ranks > 1:
                counts[1:, :2] += 11
            source, allocation, _ = select_allocation(counts, 2)
            plans, routes = [], []
            for rank in range(ranks):
                ids = torch.repeat_interleave(torch.arange(ranks * 2), source[rank])
                ids = ids[torch.randperm(ids.numel())]
                routes.append(ids)
                plans.append(compile_plan(source, allocation, ids, rank))
            for size in (1, 7, 1024):
                chunk_plans = []
                for rank, ids in enumerate(routes):
                    with patch.object(planning, 'HostControl', wraps=HostControl) as export:
                        plan = compile_plan(source, allocation, ids, rank,
                                            build_receive_order=False, chunk_rows=size)
                    export.assert_called_once()
                    self.assertIsNone(plan.sort_order)
                    chunk_plans.append(plan)
                chunks = [compile_chunks(plan, size, 1) for plan in chunk_plans]
                self.assertTrue(all('source_gradient_order' not in layout.__dict__ for layout in chunks))
                self.assertEqual(len({len(value) for value in chunks}), 1)
                for rank, plan in enumerate(plans):
                    sent = torch.cat([part.source_choices for part in chunks[rank]]) if chunks[rank] else torch.empty(0)
                    torch.testing.assert_close(sent.sort().values.long(), torch.arange(plan.dispatch_order.numel()))
                for target in range(ranks):
                    full = torch.cat([torch.arange(sum(plans[source_rank].token_send[:target]),
                                                   sum(plans[source_rank].token_send[:target + 1]))
                                      + source_rank * 100000 for source_rank in range(ranks)])
                    expected = full.index_select(0, plans[target].sort_order)
                    positions = {int(value): index for index, value in enumerate(expected)}
                    received_chunks = []
                    expert_rows = torch.empty((sum(plans[target].token_recv), 1), dtype=expected.dtype)
                    for ordinal in range(len(chunks[target])):
                        actual = self._assert_received_chunk(chunks, target, ordinal, positions,
                                                             plans[target].boundaries, size)
                        self.assertEqual(chunks[target][ordinal].spec.arena_row_span,
                                         (sum(value.numel() for value in received_chunks), actual.numel()))
                        received_chunks.append(actual)
                        store_rows(expert_rows, chunks[target][ordinal], actual[:, None])
                    actual_rows = torch.cat(received_chunks) if received_chunks else full.new_empty(0)
                    torch.testing.assert_close(actual_rows.sort().values, expected.sort().values)
                    restore = chunks[target].canonical_order
                    canonical = expert_rows if restore is None else expert_rows.index_select(0, restore)
                    torch.testing.assert_close(canonical[:, 0], expected)
                for local in chunks:
                    choices = (torch.cat([part.source_choices for part in local]) if local
                               else torch.empty(0, dtype=torch.int64))
                    torch.testing.assert_close(local.source_choices, choices)
                    torch.testing.assert_close(local.source_tokens, choices // local.top_k)
                    torch.testing.assert_close(choices.index_select(0, local.source_order),
                                               torch.arange(choices.numel()))

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_round_bucket_growth_never_exposes_truncated_splits(self):
        """Grow an underestimated bucket before consumption; complete buckets export once."""
        counts = torch.tensor([[0, 0, 0, 0], [6, 6, 6, 6]])
        _, allocation, _ = select_allocation(counts, 2)
        route = route_geometry(counts, allocation, 0, 2)
        complete = round_geometry(route, 0, 3)
        short = RoundGeometry(*(value[:1] for value in complete.tensors()))
        ids = torch.empty(0, dtype=torch.int64)
        with patch.object(planning, 'compile_round_geometry', side_effect=[short, complete]) as build:
            with patch.object(planning, 'HostControl', wraps=HostControl) as export:
                plan = compile_plan(counts, allocation, ids, 0, chunk_rows=4)
        self.assertEqual(build.call_count, 2)
        self.assertEqual(export.call_count, 2)
        self.assertEqual(len(plan.chunk_metadata.control), 3)
        self.assertEqual(sum(sum(row[2:4]) for row in plan.chunk_metadata.control), 12)
        with patch.object(planning, 'compile_round_geometry', return_value=complete):
            with patch.object(planning, 'HostControl', wraps=HostControl) as export:
                steady = compile_plan(counts, allocation, ids, 0, chunk_rows=4)
        export.assert_called_once()
        self.assertEqual(plan.chunk_metadata.control, steady.chunk_metadata.control)
        with self.assertRaisesRegex(ValueError, 'original planning call'):
            compile_chunks(steady, 8, 1)
        with self.assertRaisesRegex(ValueError, 'original planning call'):
            compile_chunks(compile_plan(counts, allocation, ids, 0), 4, 1)

    def _assert_received_chunk(self, chunks: list, target: int, ordinal: int, positions: dict[int, int],
                               boundaries: torch.Tensor, size: int) -> torch.Tensor:
        """Validate reciprocal splits, expert grouping and inverse restoration for one round."""
        local = chunks[target][ordinal]
        pieces = []
        for source_rank, source_chunks in enumerate(chunks):
            sent = source_chunks[ordinal]
            self.assertEqual(sent.spec.send_counts[target], local.spec.receive_counts[source_rank])
            begin = sum(sent.spec.send_counts[:target])
            choices = sent.source_choices[begin:begin + sent.spec.send_counts[target]]
            wire_rows = source_chunks.plan.unpermute_indices.index_select(0, choices).long()
            pieces.append(wire_rows + source_rank * 100000)
        wire = torch.cat(pieces)
        actual = wire if local.receive_order is None else wire.index_select(0, local.receive_order)
        original_positions = torch.tensor([positions[int(value)] for value in actual], dtype=torch.int64)
        groups = torch.bucketize(original_positions, boundaries, right=True)
        ends = torch.cat((local.boundaries.new_zeros(1), local.boundaries))
        expected_groups = torch.arange(local.spec.expert_slot_begin, local.spec.expert_slot_end)
        expected_groups = expected_groups.repeat_interleave(ends[1:] - ends[:-1])
        torch.testing.assert_close(groups, expected_groups)
        restored = actual if local.restore_order is None else actual.index_select(0, local.restore_order)
        torch.testing.assert_close(restored, wire)
        self.assertLessEqual(actual.numel(), size)
        return actual

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_rounds_balance_actual_sender_and_receiver_traffic(self):
        """Feature: Peer-volume chunk planning.
        Description: Split the four-rank example and the EP16 six-round workload.
        Expectation: Every round sends and receives the same rows on every rank.
        """
        for ranks, contribution, size in ((4, 1, 4), (16, 768, 4096)):
            source = torch.full((ranks, ranks * 2), contribution, dtype=torch.int64)
            allocation = torch.zeros((ranks * 2, ranks), dtype=torch.int64)
            for expert in range(ranks * 2):
                allocation[expert, expert // 2] = contribution * ranks
            for rank in range(ranks):
                ids = torch.repeat_interleave(torch.arange(ranks * 2), source[rank])
                plan = compile_plan(source, allocation, ids, rank, chunk_rows=size)
                control = plan.chunk_metadata.control
                self.assertEqual(len(control), contribution * ranks * 2 // size)
                self.assertEqual([sum(row[:ranks]) for row in control], [size] * len(control))
                self.assertEqual([sum(row[ranks:2 * ranks]) for row in control], [size] * len(control))
        traffic = torch.tensor([[3, 1], [1, 3]])
        actual = peer_round_quotas(traffic, 4, torch.tensor(2))
        expected = torch.tensor([[[2, 0], [0, 2]], [[1, 1], [1, 1]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(peer_round_quotas(traffic, 4, torch.tensor(0)), torch.zeros_like(expected))

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_values_and_all_gradients_match_independent_reference(self):
        """Feature: Explicit chunk forward/backward.
        Description: Exercise tails, tiny chunks, clips and transposed real Parameters.
        Expectation: Y, dX, FP32 dScore and both owner gradients match the scalar oracle.
        """
        for size, activation, transpose, need_scores in ((1, 'swiglu', False, True), (3, 'clipped', True, True),
                                                        (30, 'swiglu', True, True), (3, 'clipped', True, False)):
            torch.manual_seed(23)
            model = _Experts(activation, transpose)
            backend = chunked.enable_chunked_moonep(model, token_group=None, chunk_rows=size)
            runtime = configure_moonep(model)
            hidden = torch.randn(7, 4, requires_grad=True)
            ids = torch.tensor([[0, 1], [2, 0], [1, 2], [2, 1], [0, 2], [1, 0], [2, 0]])
            scores = (torch.rand(2, 7).t() if transpose else torch.rand(7, 2)).requires_grad_(need_scores)
            actual = model(hidden, ids, scores)
            expected = _reference(model, hidden, ids, scores)
            seed = torch.randn_like(actual)
            leaves = ((hidden, scores, model.gate_up, model.down) if need_scores
                      else (hidden, model.gate_up, model.down))
            observed = torch.autograd.grad(actual, leaves, seed)
            wanted = torch.autograd.grad(expected, leaves, seed)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            for value, reference in zip(observed, wanted):
                torch.testing.assert_close(value, reference, rtol=3e-5, atol=2e-6)
            self.assertFalse(backend._plans)
            runtime.finish_step()

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_replay_and_zero_rows_release_keys_and_maps(self):
        """Feature: Existing checkpoint and execution lifecycle.
        Description: Change token rows between checkpointed calls, then exercise empty and no-grad calls.
        Expectation: Chunk planning runs once per original call and all keys retire.
        """
        previous = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type('cpu')
        self.addCleanup(DefaultDeviceType.set_device_type, previous)
        model = _Experts()
        backend = chunked.enable_chunked_moonep(model, token_group=None, chunk_rows=2)
        runtime = configure_moonep(model)
        wrapped = wrap_moonep_checkpoint(model)
        with patch.object(chunked, 'compile_chunks', wraps=chunked.compile_chunks) as compile_once:
            for rows in (3, 7, 1):
                model.zero_grad(set_to_none=True)
                x = torch.randn(rows, 4, requires_grad=True)
                scores = torch.rand(rows, 2, requires_grad=True)
                ids = torch.arange(rows * 2).reshape(rows, 2) % 3
                wrapped(x, ids, scores).sum().backward()
                runtime.finish_step()
                self.assertFalse(backend._plans)
            self.assertEqual(compile_once.call_count, 3)
        empty = model(torch.empty(0, 4), torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2))
        empty.sum().backward()
        runtime.finish_step()
        with torch.no_grad():
            model(torch.ones(1, 4), torch.tensor([[0, 1]]), torch.ones(1, 2))
        runtime.finish_step()
        self.assertFalse(runtime.executions)
        self.assertFalse(backend._plans)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_last_weight_callback_precedes_last_dx_and_dispatch_is_one_ahead(self):
        """Feature: Token/compute and FSDP replica-gradient overlap.
        Description: Record dispatch, compute and weight-gradient submission on real CPU math.
        Expectation: D1 precedes C0; the prior return precedes full dW, which still precedes the final dX.
        """
        model = _Experts()
        backend = chunked.enable_chunked_moonep(model, token_group=None, chunk_rows=3)
        runtime = configure_moonep(model)
        events = []
        old_forward, old_gmm, old_exchange = chunked.expert_forward_values, chunked.grouped_values, chunked._exchange

        def _expert(*args):
            events.append('C')
            return old_forward(*args)

        def _exchange(*args):
            events.append('T')
            return old_exchange(*args)

        def _gmm(*args):
            events.append('GMM')
            return old_gmm(*args)

        def _launch(*values):
            events.append('dW')
            return local_home_gradients(*values)

        model.hsdp_scheduler = SimpleNamespace(hsdp_state=object())
        model._moonep_executor.gradient_extension = SimpleNamespace(
            start=_launch, pending=[], uses_accumulated_grad=lambda _state: False)
        with patch.object(chunked, 'expert_forward_values', side_effect=_expert),\
                patch.object(chunked, '_exchange', side_effect=_exchange),\
                patch.object(chunked, 'grouped_values', side_effect=_gmm):
            x, scores = torch.randn(4, 4, requires_grad=True), torch.rand(4, 2, requires_grad=True)
            output = model(x, torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2]]), scores)
            self.assertEqual(events[:3], ['T', 'T', 'C'])
            events.clear()
            output.sum().backward()
        self.assertEqual(events.count('dW'), 1)
        return_positions = [index for index, event in enumerate(events) if event == 'T']
        self.assertLess(return_positions[-2], events.index('dW'), 'the prior return must not wait for dW submission')
        self.assertLess(events.index('dW'), len(events) - 1 - events[::-1].index('GMM'))
        self.assertFalse(backend._plans)
        runtime.finish_step()

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_slow_merge_does_not_gate_dispatch_and_guards_returned_slot(self):
        """Feature: Caller-owned computation and bounded I/O reuse.
        Description: Use two event clocks with deliberately slow caller-side merges.
        Expectation: D2 precedes M0 completion; R2 waits M0 before overwriting its slot.
        """
        class _ClockStreams:
            def __init__(self) -> None:
                """Track independent logical completion times for each stream."""
                self.current = 'C'
                self.clock = {'C': 0., 'T': 0.}
                self.last_use = None

            @contextmanager
            def context(self) -> Iterator[None]:
                """Track the existing token submission domain."""
                previous, self.current = self.current, 'T'
                try:
                    yield
                finally:
                    self.current = previous

            def record(self) -> tuple[str, float]:
                """Publish the current stream's logical completion time."""
                return self.current, self.clock[self.current]

            def wait(self, event: Any) -> None:
                """Propagate only an explicit inter-stream dependency."""
                if event is not None:
                    self.clock[self.current] = max(self.clock[self.current], event[1])

            def protect(self, *values: Any) -> None:
                """CPU tensors stay owned by this test throughout the schedule."""

        counts = torch.tensor([[12, 12]])
        ids = torch.tensor([[0, 1]] * 12).reshape(-1)
        plan = compile_plan(counts, counts.T.contiguous(), ids, 0, chunk_rows=4)
        chunks = compile_chunks(plan, 4, 2)
        self.assertEqual(len(chunks), 6)
        hidden, scores = torch.randn(12, 4), torch.ones(12, 2)
        pool = chunked._ChunkIOPool(hidden.device)
        buffers = chunked._buffers(hidden, scores, chunks, save_source=False, pool=pool)
        streams, calls, merge_times, packs = _ClockStreams(), [], [], []

        def _exchange(output, payload, *_args):
            self.assertEqual(streams.current, 'T')
            calls.append(streams.clock['T'])
            streams.clock['T'] += 1
            output.copy_(payload)

        def _pack():
            self.assertEqual(streams.current, 'C')
            packs.append(True)
            return hidden.index_select(0, chunks.source_tokens)

        def _compute(_ordinal, _part, value):
            self.assertEqual(streams.current, 'C')
            streams.clock['C'] += 1
            return value + 1

        def _combine(part, value):
            self.assertEqual(streams.current, 'C')
            streams.clock['C'] += 10
            merge_times.append(streams.clock['C'])
            buffers.output.index_add_(0, part.source_tokens, value)

        with patch.object(chunked, '_exchange', side_effect=_exchange):
            chunked._run(chunks, streams, buffers, None, _pack, _compute, _combine, lambda: None, 'F')
        self.assertLess(calls[3], merge_times[0], 'D2 must not wait for source merge M0')
        self.assertGreaterEqual(calls[6], merge_times[0], 'R2 must not overwrite the slot consumed by M0')
        self.assertGreaterEqual(streams.clock['C'], merge_times[-1], 'the caller must observe final merge completion')
        torch.testing.assert_close(buffers.output, 2 * (hidden + 1))
        previous = buffers.receive[0].data_ptr()
        buffers = chunked._buffers(hidden, scores, chunks, save_source=False, pool=pool)
        self.assertEqual(buffers.receive[0].data_ptr(), previous)
        previous_done, first_call = streams.clock['C'], len(calls)
        streams.clock['C'] = 0.
        with patch.object(chunked, '_exchange', side_effect=_exchange):
            chunked._run(chunks, streams, buffers, None, _pack, _compute, _combine, lambda: None, 'B')
        self.assertGreaterEqual(calls[first_call], previous_done, 'a new caller must preserve old pooled readers')
        torch.testing.assert_close(buffers.output, 2 * (hidden + 1))
        self.assertEqual(len(packs), 2, 'each complete phase must gather the source only once')
        old_pair = pool.io_pair(hidden, 'receive', 2)
        old_pair[0].fill_(7)
        grown_pair = pool.io_pair(hidden, 'receive', old_pair[0].shape[0] + 5)
        self.assertNotEqual(grown_pair[0].data_ptr(), old_pair[0].data_ptr())
        torch.testing.assert_close(old_pair[0], torch.full_like(old_pair[0], 7))
        self.assertIs(pool.io_pair(hidden, 'receive', 1), grown_pair)
        device, group, domain = SimpleNamespace(type='npu'), object(), object()
        module = SimpleNamespace(Stream=lambda **_args: self.fail('must reuse the HCCL domain stream'))
        with patch.object(chunked.torch, 'get_device_module', return_value=module),\
                patch.object(chunked, 'communication_stream', return_value=domain) as lookup:
            reused = chunked._ChunkIOPool(device, group)
        self.assertIs(reused.token, domain)
        lookup.assert_called_once_with(group, device)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='dryrun', essential_mark='essential')
    def test_fused_home_gradients_preserve_microsteps_and_notify_fsdp(self):
        """Feature: FSDP gradient accumulation in chunked execution.
        Description: Exercise both owner layouts across nonempty and empty microsteps.
        Expectation: FP32 accumulators retain history and notify FSDP without BF16 home gradients.
        """
        for transposed in (False, True):
            with self.subTest(transposed=transposed):
                torch.manual_seed(81)
                model = _Experts(transposed=transposed)
                backend = chunked.enable_chunked_moonep(model, chunk_rows=3)
                runtime = configure_moonep(model)
                targets = (torch.randn_like(model.gate_up), torch.randn_like(model.down))
                wanted = tuple(value.double().clone() for value in targets)
                notifications = []
                scheduler = SimpleNamespace(hsdp_state=object())
                extension = SimpleNamespace(
                    uses_accumulated_grad=lambda _state: True,
                    accumulation_buffers=lambda _state: targets,
                    _layouts=lambda _state: ((None, transposed), (None, transposed)),
                    notify_accumulated=notifications.append, pending=[])
                model.hsdp_scheduler = scheduler
                model._moonep_executor.gradient_extension = extension
                for rows in (5, 0, 7):
                    hidden = torch.randn(rows, 4, requires_grad=True)
                    scores = torch.rand(rows, 2, requires_grad=True)
                    ids = torch.arange(rows * 2).reshape(rows, 2) % 3
                    actual = model(hidden, ids, scores)
                    seed = torch.randn_like(actual)
                    if rows:
                        reference = _reference(model, hidden, ids, scores)
                        expected = torch.autograd.grad(reference, (hidden, scores, model.gate_up, model.down), seed)
                        wanted = tuple(previous + value.double() for previous, value in zip(wanted, expected[2:]))
                    else:
                        expected = (torch.zeros_like(hidden), torch.zeros_like(scores))
                    with patch('hyper_parallel.distributed.expert_parallel.moonep.runtime.grouped_weight_grad_add',
                               wraps=runtime_module.grouped_weight_grad_add) as add:
                        actual.backward(seed)
                    self.assertEqual(add.call_count, 2, f'add calls={add.call_count}, expected=2')
                    torch.testing.assert_close(hidden.grad, expected[0], rtol=3e-5, atol=2e-6)
                    torch.testing.assert_close(scores.grad, expected[1], rtol=3e-5, atol=2e-6)
                    for target, reference in zip(targets, wanted):
                        self.assertEqual(target.dtype, torch.float32)
                        torch.testing.assert_close(target.double(), reference, rtol=3e-5, atol=2e-6)
                    self.assertIsNone(model.gate_up.grad)
                    self.assertIsNone(model.down.grad)
                    self.assertFalse(backend._plans)
                    runtime.finish_step()
                self.assertEqual(notifications, [scheduler] * 3)
                self.assertEqual(runtime.statistics['home_wgrad_add_calls'], 6)
