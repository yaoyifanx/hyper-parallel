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
"""CPU replica pool ownership checks with mocked NPU streams and collectives."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch
import weakref

import torch

from hyper_parallel.core.multicore.modules.routed_experts.arena import PhaseArena
from hyper_parallel.distributed.expert_parallel.moonep.fused import weights as weights_module
from hyper_parallel.distributed.expert_parallel.moonep.fused.pool import ReplicaPool, overlaps, prepare_gradient_storage
from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import (
    Fragment, ReplicaTransfer, compile_replica_transfer,
)
from hyper_parallel.distributed.expert_parallel.moonep.fused.weights import TransferDomain
from tests.common.mark_utils import arg_mark


class _Streams:
    """Represent stream/event submission order without any device initialization."""

    def __init__(self) -> None:
        """Create a compute stream and collect communication events for inspection."""
        self.compute = Mock(name="compute_stream")
        self.current = self.compute
        self.events = []

    @staticmethod
    def Stream(**_kwargs: Any) -> Mock:  # pylint: disable=invalid-name
        """Create a communication stream handle with observable wait calls."""
        return Mock(name="communication_stream")

    def Event(self) -> Mock:  # pylint: disable=invalid-name
        """Create a fence whose record stream can be inspected by the test."""
        event = Mock(name="event")
        self.events.append(event)
        return event

    def current_stream(self, _device: torch.device) -> Mock:
        """Return the current mocked stream."""
        return self.current

    @contextmanager
    def stream(self, stream: Mock) -> Iterator[None]:
        """Run a callback inside its intended communication stream."""
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestReplicaPool(unittest.TestCase):
    """Check parameterized byte accounting and physical matrix layouts."""

    def test_parameterized_budget_uses_replica_capacity(self):
        """Weight slots have explicit byte bounds; gradient domains retain geometry only."""
        shapes = ((2048, 1536), (768, 2048))
        for capacity in (0, 1, 8, 16):
            for slots in (2, 3):
                with self.subTest(capacity=capacity, slots=slots):
                    weight = ReplicaPool(torch.device("meta"), torch.bfloat16, slot_count=slots)
                    gradient = ReplicaPool(torch.device("meta"), torch.float32)
                    weight.allocate(8, capacity, shapes, (True, True))
                    gradient.allocate(8, capacity, shapes, (True, True))
                    self.assertEqual(len(weight.slots), slots)
                    self.assertEqual(weight.reserved_bytes, 0)
                    for slot in weight.slots:
                        weight.ensure_storage(slot, capacity)
                    self.assertEqual(weight.reserved_bytes, slots * capacity * 9 * 1024**2)
                    self.assertEqual(gradient.reserved_bytes, 0)


    def test_incoming_capacity_covers_empty_low_and_large_token_counts(self):
        """Early G writes preserve every dV row still needed by deferred home dW."""
        for rows in (0, 1, 16):
            with self.subTest(rows=rows):
                arena = PhaseArena("cpu")
                token = arena.begin()
                incoming, staging = prepare_gradient_storage(
                    arena.allocate, (rows, 4), staging_elements=(16, 16))
                incoming.fill_(9)
                self.assertEqual(incoming.shape, (rows, 4))
                for value in staging:
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertEqual(value.shape, (16,))
                    self.assertEqual(value.data_ptr() % 32, 0)
                    self.assertFalse(overlaps(incoming, value))
                    value.fill_(7)
                self.assertFalse(overlaps(*staging))
                torch.testing.assert_close(incoming, torch.full_like(incoming, 9))
                pointer = incoming.untyped_storage().data_ptr()
                arena.end(token)
                token = arena.begin()
                again, _ = prepare_gradient_storage(arena.allocate, (rows, 4), staging_elements=(16, 16))
                self.assertEqual(again.untyped_storage().data_ptr(), pointer)
                arena.end(token)
                arena.close()


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestTransferDomainPool(unittest.TestCase):
    """Check leases and complete gradient outputs without a process group or NPU."""

    def setUp(self) -> None:
        """Replace all stream, storage-recording, and communication boundaries."""
        self.streams = _Streams()
        self.distributed = Mock()
        self.distributed.get_world_size.return_value = 2
        self.distributed.all_to_all_single.side_effect = self._exchange
        self.work = Mock()
        patches = (patch.object(torch.Tensor, "record_stream"),
                   patch.object(weights_module.torch, "npu", self.streams, create=True),
                   patch.object(weights_module, "dist", self.distributed))
        started = []
        for patcher in patches:
            started.append(patcher.start())
            self.addCleanup(patcher.stop)
        self.record = started[0]
        self.native = self.enterContext(patch.object(weights_module, "prepare_ordered_sum",
                                                    side_effect=self._ordered_sum))
        self.shapes, self.layouts = ((2, 4), (2, 2)), (False, False)
        self.home = (torch.ones((1, 2, 4), dtype=torch.bfloat16), torch.ones((1, 2, 2), dtype=torch.bfloat16))
        allocation = torch.tensor([[1, 0], [1, 1]], dtype=torch.int64)
        self.transfer = compile_replica_transfer(allocation, (("gate", 8), ("down", 4)))

    def _exchange(self, output, _input, _receive, _send, **_kwargs):
        output.fill_(7)
        return self.work

    def _domain(self, dtype=torch.bfloat16, *, admit=True):
        domain = TransferDomain(object(), 0, torch.device("cpu"), dtype=dtype,
                                stream=self.streams.Stream())
        if admit:
            domain.admit(1, 1, self.shapes, self.layouts)
        return domain

    def _materialize(self, domain, *, remote=True):
        def _restore():
            self.assertIs(self.streams.current, domain.stream)
            return self.home

        return domain.materialize((0, 1) if remote else (0,), 1, _restore,
                                  self.transfer if remote else None, self.shapes, self.layouts)

    def test_weight_fragments_use_matching_contiguous_physical_layouts(self):
        """Send physical bytes even for submatrix packets; return the original logical weights."""
        for layouts in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(layouts=layouts):
                self.layouts = layouts
                values = (torch.arange(8, dtype=torch.bfloat16).view(1, 2, 4),
                          torch.arange(16, 20, dtype=torch.bfloat16).view(1, 2, 2))
                self.home = tuple(value.transpose(-2, -1).contiguous().transpose(-2, -1) if transpose else value
                                  for value, transpose in zip(values, layouts))
                self.transfer = compile_replica_transfer(torch.ones(2, 2, dtype=torch.int64),
                                                     (("gate", 8), ("down", 4)))
                packets = []

                def exchange(output: torch.Tensor, payload: torch.Tensor, *_args: Any, **_kwargs: Any) -> Mock:
                    """Capture and mirror a packet without performing communication.

                    Args:
                        output: Borrowed receive buffer.
                        payload: Physical wire bytes selected for this round.
                        *_args: Unused collective positional arguments.
                        **_kwargs: Unused collective keyword arguments.

                    Returns:
                        Mock communication work handle.
                    """
                    packets.append(payload.clone())
                    output.copy_(payload)
                    return self.work

                self.distributed.all_to_all_single.side_effect = exchange
                domain = TransferDomain(object(), 0, torch.device("cpu"),
                                        stream=self.streams.Stream())
                domain.admit(1, 1, self.shapes, layouts)
                self.record.reset_mock()
                lease = self._materialize(domain)
                # Borrowed home/remote weights cross streams; private EP
                # send/receive staging must not acquire redundant stream uses.
                self.assertEqual(self.record.call_count, 4)
                for role, expected in zip(("gate", "down"), values):
                    torch.testing.assert_close(lease.remote[1, role], expected[0])
                physical = [value.transpose(-2, -1) if transpose else value
                            for value, transpose in zip(self.home, layouts)]
                expected_wire = torch.cat((physical[1].flatten(), physical[0].flatten()))
                torch.testing.assert_close(torch.cat(packets), expected_wire)
                domain.release_lease(lease)

    def test_slot_reuse_waits_for_last_compute_reader(self):
        """A released slot reuses storage only after W waits for its C completion fence."""
        domain = self._domain()
        for slot in domain._pool.slots:
            domain._pool.ensure_storage(slot, 1)
        first = self._materialize(domain)
        with patch.object(weights_module.torch, "empty", side_effect=AssertionError("hot allocation")):
            second = self._materialize(domain)
            self.assertTrue(all(slot.in_use for slot in domain._pool.slots))
            self.assertNotEqual(first.remote[1, "gate"].data_ptr(), second.remote[1, "gate"].data_ptr())
            domain.release_lease(first)
            last_reader = self.streams.events[-1]
            self.streams.compute.wait_event.assert_called_with(first.ready)
            last_reader.record.assert_called_once_with(self.streams.compute)
            third = self._materialize(domain)
            domain.stream.wait_event.assert_called_with(last_reader)
            self.assertEqual(first.remote[1, "gate"].data_ptr(), third.remote[1, "gate"].data_ptr())
            self.assertEqual(third.generation, first.generation + 1)
            domain.release_lease(second)
            domain.release_lease(third)
        self.assertTrue(all(not slot.in_use for slot in domain._pool.slots))
        self.assertTrue(self.record.called)
        self.assertEqual(self.distributed.all_reduce.call_count, 3)
        for call in self.distributed.all_reduce.call_args_list:
            self.assertIs(call.kwargs["group"], domain.control_group)
        self.assertEqual(self.work.wait.call_count, 3)

    def test_zero_remote_still_consumes_a_logical_slot(self):
        """Empty receivers cannot advance their collective schedule past two leases."""
        domain = self._domain()
        first, second = self._materialize(domain, remote=False), self._materialize(domain, remote=False)
        self.assertEqual(first.remote, {})
        self.assertEqual(second.remote, {})
        self.assertTrue(all(slot.in_use for slot in domain._pool.slots))
        domain.release_lease(first)
        self.assertTrue(any(not slot.in_use for slot in domain._pool.slots))
        domain.release_lease(second)
        self.distributed.all_to_all_single.assert_not_called()

    def test_global_transfer_keeps_zero_split_participants(self) -> None:
        """A migration between peers 1/2 still requires rank 0's zero-split collective."""
        self.distributed.get_world_size.return_value = 3
        domain = self._domain()
        allocation = torch.eye(3, dtype=torch.int64)
        allocation[1, 2] = 1
        transfer = compile_replica_transfer(allocation, (("gate", 8), ("down", 4)))
        self.assertIs(domain._prepare_transfer(transfer, (0,)), transfer.local(0))

        lease = domain.materialize((0,), 1, lambda: self.home, transfer, self.shapes, self.layouts)

        self.distributed.all_to_all_single.assert_called_once()
        self.assertEqual(self.distributed.all_to_all_single.call_args.args[2:4], ([0, 0, 0], [0, 0, 0]))
        self.work.wait.assert_called_once()
        self.assertEqual(lease.remote, {})
        domain.release_lease(lease)

    def test_invalid_single_transfer_is_rejected_before_collective(self) -> None:
        """None coverage, ownership, duplicates, bounds and peer order retain their errors."""
        transfer = compile_replica_transfer(torch.ones(2, 2, dtype=torch.int64), (("gate", 8), ("down", 4)))
        fragments = transfer.fragments
        first = fragments[0]
        invalid = (
            (None, "completely cover"),
            (ReplicaTransfer(fragments, 3), "communication group"),
            (ReplicaTransfer((*fragments, fragments[-1]), 2), "exactly once"),
            (ReplicaTransfer(fragments[:-1], 2), "completely cover"),
            (ReplicaTransfer((Fragment(0, 1, 1, first.role, first.elements), *fragments[1:]), 2), "ownership"),
            (ReplicaTransfer((*fragments, Fragment(3, 0, 0, first.role, first.elements)), 2), "peer/expert"),
            (ReplicaTransfer((Fragment(0, 1, 0, first.role, first.elements + 1), *fragments[1:]), 2), "matrix bounds"),
            (ReplicaTransfer((Fragment(0, 1, 0, "invalid", first.elements), *fragments[1:]), 2), "gate or down"),
        )
        for descriptor, message in invalid:
            with self.subTest(message=message):
                domain = self._domain()
                with self.assertRaisesRegex(ValueError, message):
                    domain.materialize((0, 1), 1, lambda: self.home, descriptor, self.shapes, self.layouts)
        domain = self._domain(torch.float32)
        with self.assertRaisesRegex(ValueError, "ownership"):
            domain._prepare_transfer(transfer, (0, 1), reverse=True)
        self.distributed.get_world_size.return_value = 3
        domain = self._domain()
        unsorted = ReplicaTransfer((Fragment(0, 2, 0, "gate", 8), Fragment(0, 1, 0, "gate", 8)), 3)
        with self.assertRaisesRegex(ValueError, "peer order"):
            domain._prepare_transfer(unsorted, (0,))
        self.distributed.all_to_all_single.assert_not_called()


    @staticmethod
    def _ordered_sum(groups, targets, local):
        """Keep source addresses unready until the same ordered FP32 reduction runs."""
        def run() -> None:
            """Evaluate the preserved local-first, then peer-ordered FP32 sum."""
            for sources, target, present in zip(groups, targets, local):
                value = sources[0].clone() if present else torch.zeros_like(target, dtype=torch.float32)
                for source in sources[int(present):]:
                    value.add_(source)
                target.copy_(value)
        return SimpleNamespace(run=run)

    def test_early_gradient_payload_preserves_dv_until_compute_completion(self):
        """Wire remote dW early, then consume the later local dW exactly once on C."""
        domain = self._domain(torch.float32)
        gradients = (torch.full((2, 2, 4), 1.25), torch.full((2, 2, 2), 2.5))
        outputs = tuple(torch.full_like(value[:1], -1, dtype=torch.bfloat16) for value in gradients)
        transfer = compile_replica_transfer(torch.ones((2, 2), dtype=torch.int64),
                                          (("gate", 8), ("down", 4)), reverse=True)
        incoming, staging = prepare_gradient_storage(
            lambda _name, shape, dtype: torch.empty(shape, dtype=dtype), (16, 4), staging_elements=(12, 12))
        incoming.fill_(9)
        producer = Mock()
        packet = domain.prepare_gradient_payload((0, 1), 1, gradients, transfer, producer,
                                                 outputs=outputs, staging=staging)
        domain.stream.wait_stream.reset_mock()
        domain.submit_gradient_payload(packet)
        producer.wait_on_stream.assert_called_once_with(domain.stream)
        domain.stream.wait_stream.assert_not_called()
        torch.testing.assert_close(incoming, torch.full_like(incoming, 9))
        for output in outputs:
            torch.testing.assert_close(output, torch.full_like(output, -1))
        gradients[0][0].fill_(3.25)
        gradients[1][0].fill_(4.5)
        self.streams.compute.wait_event.reset_mock()
        completed = domain.complete_gradients(packet)
        self.native.assert_called_once()
        self.streams.compute.wait_event.assert_called_once_with(packet.ready)
        self.distributed.all_to_all_single.assert_called_once()
        self.assertTrue(packet.finished)
        for output, expected in zip(completed, (10.25, 11.5)):
            torch.testing.assert_close(output, torch.full_like(output, expected))
        self.assertEqual(domain.reserved_bytes, 0)
        with self.assertRaisesRegex(ValueError, "already been completed"):
            domain.complete_gradients(packet)
        self.assertTrue(domain.poisoned)

    def test_gradient_output_cannot_alias_payload_or_domain_storage(self):
        """Different dtypes cannot hide a writable buffer alias before admission."""
        transfer = compile_replica_transfer(torch.ones((2, 2), dtype=torch.int64),
                                          (("gate", 8), ("down", 4)), reverse=True)
        for owner in ("gradient", "send", "receive"):
            with self.subTest(owner=owner):
                domain = self._domain(torch.float32)
                gradients = (torch.ones((2, 2, 4)), torch.ones((2, 2, 2)))
                staging = (torch.empty(12), torch.empty(12))
                storage = gradients[0] if owner == "gradient" else staging[owner == "receive"]
                outputs = (storage.view(torch.bfloat16).reshape(-1)[:8].view(1, 2, 4),
                           torch.empty((1, 2, 2), dtype=torch.bfloat16))
                with self.assertRaisesRegex(ValueError, "must not alias"):
                    domain.prepare_gradient_payload((0, 1), 1, gradients, transfer, Mock(),
                                                    outputs=outputs, staging=staging)
        self.distributed.all_to_all_single.assert_not_called()

    def test_empty_gradient_frame_does_not_retain_phase_storage(self):
        """A no-migration phase writes positive zero and releases borrowed staging."""
        domain = self._domain(torch.float32)
        arena = PhaseArena("cpu")
        token = arena.begin()
        incoming, staging = prepare_gradient_storage(arena.allocate, (0, 4), staging_elements=(0, 0))
        reference = weakref.ref(staging[0]._base)
        gradients = (torch.empty((0, 2, 4)), torch.empty((0, 2, 2)))
        outputs = tuple(torch.full((1, *value.shape[1:]), -1, dtype=torch.bfloat16) for value in gradients)
        packet = domain.prepare_gradient_payload((), 1, gradients, None, Mock(), outputs=outputs, staging=staging)
        domain.submit_gradient_payload(packet)
        self.assertIs(domain.complete_gradients(packet), outputs)
        for value in outputs:
            torch.testing.assert_close(value, torch.zeros_like(value), rtol=0, atol=0)
        self.distributed.all_to_all_single.assert_not_called()
        arena.end(token)
        arena.release_regions(("incoming",))
        del incoming, staging, packet
        self.assertIsNone(reference())
        self.assertEqual(domain.buffers, {})
        arena.close()
