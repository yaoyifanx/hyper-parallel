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
"""CPU admission/legacy fallbacks for allocation-free short FD dictionaries."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hyper_parallel.core.multicore.modules.routed_experts import transport
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestCompactFDState(unittest.TestCase):
    """Keep older payloads safe and avoid memset when native UB admission succeeds."""

    def setUp(self) -> None:
        self.workspace = SimpleNamespace(spec=SimpleNamespace(ranks=2, block_rows=16),
                                         device=torch.device('cpu'), metadata={})

    def test_legacy_payload_keeps_both_zeroed_planes(self) -> None:
        """Older libraries retain their original state and group contract."""
        with mock.patch.object(transport, 'compact_fd_available', return_value=False):
            state = transport._allocate_dedup_state(64, self.workspace, False, .75)
            self.assertTrue(transport.requires_forward_groups(0.0))
        self.assertEqual(state.shape, (2, 2, 64))
        self.assertEqual(state.dtype, torch.int64)
        self.assertEqual(torch.count_nonzero(state).item(), 0)

    def test_resident_dictionary_uses_tag_without_zero_kernel(self) -> None:
        """The only GM object is a cache-line tag; no initialization is queued."""
        with mock.patch.object(transport, 'compact_fd_available', return_value=True), mock.patch.object(
                transport, '_dedup_state_words', return_value=8), mock.patch.object(
                transport, 'transport_worker_count', return_value=4), mock.patch.object(
                transport.torch, 'zeros', side_effect=AssertionError('resident dictionary must not clear GM')):
            state = transport._allocate_dedup_state(64, self.workspace, False, .75)
            self.assertFalse(transport.requires_forward_groups(0.0))
            self.assertTrue(transport.requires_forward_groups(1.0))
        self.assertEqual(state.shape, (8,))
        self.assertEqual(state.dtype, torch.int64)

    def test_resident_tag_is_shared_across_invocations(self) -> None:
        """Different admitted source extents reuse one immutable tag storage."""
        with mock.patch.object(transport, 'compact_fd_available', return_value=True), mock.patch.object(
                transport, '_dedup_state_words', return_value=8), mock.patch.object(
                transport, 'transport_worker_count', return_value=4):
            first = transport._allocate_dedup_state(64, self.workspace, False, .75)
            second = transport._allocate_dedup_state(128, self.workspace, False, .75)
        self.assertIs(first, second)
        self.assertEqual(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr())

    def test_cached_tag_does_not_replace_full_gm_dictionary(self) -> None:
        """A later nonresident shape still receives its own fully zeroed state."""
        with mock.patch.object(transport, 'compact_fd_available', return_value=True), mock.patch.object(
                transport, '_dedup_state_words', side_effect=[8, 128]), mock.patch.object(
                transport, 'transport_worker_count', return_value=4):
            tag = transport._allocate_dedup_state(64, self.workspace, False, .75)
            full = transport._allocate_dedup_state(65536, self.workspace, False, .75)
        self.assertIsNot(tag, full)
        self.assertEqual(full.shape, (128,))
        self.assertEqual(torch.count_nonzero(full).item(), 0)

    def test_nonresident_dictionary_is_fully_initialized(self) -> None:
        """Long and multiplexed cases retain the admitted zeroed GM table."""
        with mock.patch.object(transport, 'compact_fd_available', return_value=True), mock.patch.object(
                transport, '_dedup_state_words', return_value=128), mock.patch.object(
                transport, 'transport_worker_count', return_value=4):
            state = transport._allocate_dedup_state(64, self.workspace, True, .75)
        self.assertEqual(state.shape, (128,))
        self.assertEqual(torch.count_nonzero(state).item(), 0)

    def test_malformed_native_capacity_is_rejected(self) -> None:
        """A malformed admission cannot silently create an undersized state."""
        with mock.patch.object(transport, 'compact_fd_available', return_value=True), mock.patch.object(
                transport, '_dedup_state_words', return_value=7), mock.patch.object(
                transport, 'transport_worker_count', return_value=4):
            with self.assertRaisesRegex(RuntimeError, 'complete cache lines'):
                transport._allocate_dedup_state(64, self.workspace, False, .75)

    def test_one_initialization_fence_follows_dictionary_initialization(self) -> None:
        """Counted FD must cover GM initialization without re-recording an event."""
        order = []
        events = []

        class Event:
            def __init__(self):
                self.records = 0
                events.append(self)

            def record(self, _stream):
                self.records += 1
                order.append('record')

        def initialize(*_args):
            order.append('table')
            return torch.empty(8, dtype=torch.long)

        source = torch.zeros(2, 16, dtype=torch.bfloat16)
        empty = torch.empty(0, dtype=torch.long)
        workspace = SimpleNamespace(closed=False, poisoned=False, closing=False, metadata={})
        route = SimpleNamespace(received_rows=2, outgoing=())
        values = (torch.empty_like(source), empty, empty, empty, empty, (), empty, object())
        with mock.patch.object(transport, 'requires_forward_groups', return_value=False), mock.patch.object(
                transport, '_prepare', return_value=values), mock.patch.object(
                transport, '_prepare_readiness', return_value=(None, None)), mock.patch.object(
                transport, '_check_dependency'), mock.patch.object(
                transport, '_allocate_dedup_state', side_effect=initialize), mock.patch.object(
                transport.torch, 'npu', SimpleNamespace(Event=Event), create=True):
            prepared = transport.prepare_exchange(source, route, workspace, mapping=empty,
                                                 dedup=True, group_metadata=None, reduce_dense_ratio=0.)
        self.assertEqual(order, ['table', 'record'])
        self.assertEqual(prepared.initialized.records, 1)
