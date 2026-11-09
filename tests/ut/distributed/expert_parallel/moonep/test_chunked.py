# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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
from hyper_parallel.distributed.expert_parallel.moonep import chunked
from hyper_parallel.distributed.expert_parallel.moonep.chunked_plan import chunk_specs, compile_chunks
from hyper_parallel.distributed.expert_parallel.moonep.planning import compile_plan, select_allocation
from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPExecutor, local_home_gradients
from tests.common.mark_utils import arg_mark


class _Experts(torch.nn.Module):
    def __init__(self, activation: str = 'swiglu', transposed: bool = False) -> None:
        """Create a small expert owner supporting either physical weight layout."""
        super().__init__()
        self.transposed = transposed
        gate, down = torch.randn(3, 4, 6) * .2, torch.randn(3, 3, 4) * .2
        self.gate = torch.nn.Parameter(gate.transpose(1, 2).contiguous() if transposed else gate)
        self.down = torch.nn.Parameter(down.transpose(1, 2).contiguous() if transposed else down)
        self._moonep_executor = MoonEPExecutor(self, self.weights, ep_group=None, ep_size=1,
                                              num_experts=3, top_k=2, activation=activation, swiglu_limit=.25)

    def weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logical views while retaining the real Parameter edges."""
        return ((self.gate.transpose(1, 2), self.down.transpose(1, 2)) if self.transposed
                else (self.gate, self.down))

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Execute the same owner boundary used by model adapters."""
        return self._moonep_executor(hidden, ids, scores, expert_weights=self.weights())


def _reference(model, hidden, ids, scores):
    gate, down = model.weights()
    result = torch.zeros_like(hidden)
    for token in range(hidden.shape[0]):
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            left, right = (hidden[token] @ gate[expert]).chunk(2)
            if model._moonep_executor.activation == 'clipped':
                left, right = left.clamp(max=.25), right.clamp(-.25, .25)
            result[token] = result[token] + scores[token, choice] * ((functional.silu(left) * right) @ down[expert])
    return result


class TestChunkedMoonEP(unittest.TestCase):
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
            plans = []
            for rank in range(ranks):
                ids = torch.repeat_interleave(torch.arange(ranks * 2), source[rank])
                ids = ids[torch.randperm(ids.numel())]
                plans.append(compile_plan(source, allocation, ids, rank))
            for size in (1, 7, 1024):
                chunks = [compile_chunks(plan, rank, size, 1) for rank, plan in enumerate(plans)]
                self.assertEqual(len({len(value) for value in chunks}), 1)
                for rank, plan in enumerate(plans):
                    sent = torch.cat([part.send_indices for part in chunks[rank]]) if chunks[rank] else torch.empty(0)
                    torch.testing.assert_close(sent.sort().values.long(), torch.arange(plan.dispatch_order.numel()))
                for target in range(ranks):
                    full = torch.cat([torch.arange(sum(plans[source_rank].token_send[:target]),
                                                   sum(plans[source_rank].token_send[:target + 1]))
                                      + source_rank * 100000 for source_rank in range(ranks)])
                    expected = full.index_select(0, plans[target].sort_order)
                    positions = {int(value): index for index, value in enumerate(expected)}
                    received_chunks = []
                    for ordinal in range(len(chunks[target])):
                        actual = self._assert_received_chunk(chunks, target, ordinal, positions,
                                                             plans[target].boundaries, size)
                        received_chunks.append(actual)
                    actual_rows = torch.cat(received_chunks) if received_chunks else full.new_empty(0)
                    torch.testing.assert_close(actual_rows.sort().values, expected.sort().values)
                for local in chunks:
                    choices = (torch.cat([part.source_choices for part in local]) if local
                               else torch.empty(0, dtype=torch.int64))
                    torch.testing.assert_close(choices.index_select(0, local.source_order),
                                               torch.arange(choices.numel()))

    def _assert_received_chunk(self, chunks: list, target: int, ordinal: int, positions: dict[int, int],
                               boundaries: torch.Tensor, size: int) -> torch.Tensor:
        """Validate reciprocal splits, expert grouping and inverse restoration for one round."""
        local = chunks[target][ordinal]
        pieces = []
        for source_rank, source_chunks in enumerate(chunks):
            sent = source_chunks[ordinal]
            self.assertEqual(sent.spec.send_counts[target], local.spec.receive_counts[source_rank])
            begin = sum(sent.spec.send_counts[:target])
            pieces.append(sent.send_indices[begin:begin + sent.spec.send_counts[target]] + source_rank * 100000)
        wire = torch.cat(pieces)
        actual = wire if local.receive_order is None else wire.index_select(0, local.receive_order)
        original_positions = torch.tensor([positions[int(value)] for value in actual], dtype=torch.int64)
        groups = torch.bucketize(original_positions, boundaries, right=True)
        ends = torch.tensor((0, *local.spec.boundaries), dtype=torch.int64)
        expected_groups = torch.arange(local.spec.weight_begin, local.spec.weight_end)
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
                plan = SimpleNamespace(source_counts=source, allocation=allocation,
                                       token_send=(contribution * 2,) * ranks)
                specs = chunk_specs(plan, rank, size)
                self.assertEqual(len(specs), contribution * ranks * 2 // size)
                self.assertEqual([sum(spec.send_counts) for spec in specs], [size] * len(specs))
                self.assertEqual([sum(spec.receive_counts) for spec in specs], [size] * len(specs))

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
            leaves = ((hidden, scores, model.gate, model.down) if need_scores
                      else (hidden, model.gate, model.down))
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

        def _launch(values):
            events.append('dW')
            return local_home_gradients(*values)

        model.hsdp_scheduler = SimpleNamespace(launch_gradient_extension=_launch)
        with patch.object(chunked, 'expert_forward_values', side_effect=_expert), \
                patch.object(chunked, '_exchange', side_effect=_exchange), \
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
        """Feature: Independent source merge with bounded I/O reuse.
        Description: Use a deterministic event-clock model with deliberately slow merges.
        Expectation: D2 precedes M0 completion; R2 waits M0 before overwriting its slot.
        """
        class _ClockStreams:
            def __init__(self) -> None:
                """Track independent logical completion times for each stream."""
                self.current = 'C'
                self.clock = {'C': 0., 'T': 0., 'M': 0.}
                self.last_merge = None

            @contextmanager
            def context(self, *, merge: bool = False) -> Iterator[None]:
                """Track independent token and merge submission domains."""
                previous, self.current = self.current, 'M' if merge else 'T'
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
        plan = compile_plan(counts, counts.T.contiguous(), ids, 0)
        chunks = compile_chunks(plan, 0, 4, 2)
        self.assertEqual(len(chunks), 6)
        hidden, scores = torch.randn(12, 4), torch.ones(12, 2)
        pool = chunked._Streams(hidden.device)
        buffers = chunked._buffers(hidden, scores, chunks, save_source=False, score_gradient=False, pool=pool)
        streams, calls, merge_times = _ClockStreams(), [], []

        def _exchange(output, payload, *_args):
            calls.append(streams.clock['T'])
            streams.clock['T'] += 1
            output.copy_(payload)

        def _pack(part, payload):
            payload.copy_(hidden.index_select(0, part.source_tokens))

        def _compute(_ordinal, _part, value):
            streams.clock['C'] += 1
            return value + 1

        def _combine(part, value):
            streams.clock['M'] += 10
            merge_times.append(streams.clock['M'])
            buffers.output.index_add_(0, part.source_tokens, value)

        with patch.object(chunked, '_exchange', side_effect=_exchange):
            chunked._run(chunks, streams, buffers, None, _pack, _compute, _combine, lambda: None, 'F')
        self.assertLess(calls[3], merge_times[0], 'D2 must not wait for source merge M0')
        self.assertGreaterEqual(calls[6], merge_times[0], 'R2 must not overwrite the slot consumed by M0')
        self.assertGreaterEqual(streams.clock['C'], merge_times[-1], 'the caller must observe final merge completion')
        torch.testing.assert_close(buffers.output, 2 * (hidden + 1))
        previous = buffers.send[0].data_ptr()
        buffers = chunked._buffers(hidden, scores, chunks, save_source=False, score_gradient=False, pool=pool)
        self.assertEqual(buffers.send[0].data_ptr(), previous)
        previous_done, first_call = streams.clock['M'], len(calls)
        streams.clock['C'] = 0.
        with patch.object(chunked, '_exchange', side_effect=_exchange):
            chunked._run(chunks, streams, buffers, None, _pack, _compute, _combine, lambda: None, 'B')
        self.assertGreaterEqual(calls[first_call], previous_done, 'a new caller must preserve old pooled readers')
        torch.testing.assert_close(buffers.output, 2 * (hidden + 1))
        old_pair = pool.io_pair(hidden, 'send', 2)
        old_pair[0].fill_(7)
        grown_pair = pool.io_pair(hidden, 'send', old_pair[0].shape[0] + 5)
        self.assertNotEqual(grown_pair[0].data_ptr(), old_pair[0].data_ptr())
        torch.testing.assert_close(old_pair[0], torch.full_like(old_pair[0], 7))
        self.assertIs(pool.io_pair(hidden, 'send', 1), grown_pair)
