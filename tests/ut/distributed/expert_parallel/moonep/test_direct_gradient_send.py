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
"""Direct FP32 replica output ownership, wire order, and forbidden aliases."""
from types import SimpleNamespace
import unittest

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fused.transfers import compile_replica_transfer
from hyper_parallel.distributed.expert_parallel.moonep.fused.weights import TransferDomain


class TestDirectGradientSend(unittest.TestCase):
    """Exercise three peer destinations independently of active expert ordering."""

    def setUp(self) -> None:
        """Prepare physical owner matrices and three peer-ordered send regions."""
        self.domain = TransferDomain.__new__(TransferDomain)
        self.domain.admitted, self.domain.parked = True, False
        self.domain.dtype, self.domain.device = torch.float32, torch.device("cpu")
        self.domain.rank, self.domain._ranks = 0, 4
        self.domain._pool = SimpleNamespace(home_count=2, shapes=((4, 8), (4, 4)), transposed=(True, False))
        self.outputs = (torch.full((2, 8, 4), 17.0), torch.full((2, 4, 4), 29.0))
        self.active = (6, 0, 4, 3)
        allocation = torch.zeros(8, 4, dtype=torch.int64)
        for expert in range(8):
            allocation[expert, expert // 2] = 2
        allocation[[3, 4, 6], 0] = 1
        allocation[0, 2] = 1
        self.transfer = compile_replica_transfer(allocation, (("gate", 32), ("down", 16)), reverse=True)
        self.local = self.transfer.local(0)
        self.staging = (torch.empty(sum(self.local.send_sizes)), torch.empty(sum(self.local.receive_sizes)))

    def _targets(self):
        return self.domain.accumulation_targets(
            self.active, 2, self.outputs, self.transfer, staging=self.staging)

    def test_direct_producers_match_independent_peer_ordered_packet(self):
        """Interleaved down/gate fragments must produce the established wire layout."""
        targets = self._targets()
        reference = {}
        for role in range(2):
            for expert, target in zip(self.active, targets[role]):
                value = torch.arange(target.numel()).view_as(target).float() + 100 * expert + role
                reference[expert, role] = value
                if expert // 2 != 0:
                    target.copy_(value)
        expected = torch.cat([reference[fragment.expert, fragment.role == "down"].flatten()
                              for fragment in self.local.sent])
        torch.testing.assert_close(self.staging[0], expected, rtol=0, atol=0)
        for value, initial in zip(self.outputs, (17.0, 29.0)):
            torch.testing.assert_close(value, torch.full_like(value, initial), rtol=0, atol=0)
        packet = self.domain.prepare_accumulated_payload(
            self.active, 2, targets, self.transfer, object(), outputs=self.outputs,
            staging=self.staging, direct_send=True)
        self.assertTrue(packet.direct_send)
        self.assertIs(packet.buffers[0], self.staging[0])

    def test_nearby_send_slice_is_rejected(self):
        """Same storage and shape do not authorize writing another fragment's bytes."""
        targets = [list(role) for role in self._targets()]
        original = targets[0][0]
        targets[0][0] = self.staging[0][1:1 + original.numel()].view_as(original)
        with self.assertRaisesRegex(ValueError, "exact owner or peer-ordered send slice"):
            self.domain.prepare_accumulated_payload(
                self.active, 2, targets, self.transfer, object(), outputs=self.outputs,
                staging=self.staging, direct_send=True)

    def test_receive_storage_cannot_be_a_remote_output(self):
        """Receive storage may be written by HCCL while native producers run."""
        targets = [list(role) for role in self._targets()]
        original = targets[0][0]
        targets[0][0] = self.staging[1][:original.numel()].view_as(original)
        with self.assertRaisesRegex(ValueError, "exact owner or peer-ordered send slice"):
            self.domain.prepare_accumulated_payload(
                self.active, 2, targets, self.transfer, object(), outputs=self.outputs,
                staging=self.staging, direct_send=True)

    def test_owner_storage_cannot_overlap_staging(self):
        """Sending real accumulated g would include earlier microstep contributions."""
        outputs = (self.staging[0][:64].view(2, 8, 4), self.outputs[1])
        with self.assertRaisesRegex(ValueError, "owner gradients|owner gate"):
            self.domain.accumulation_targets(self.active, 2, outputs, self.transfer, staging=self.staging)

    def test_legacy_payload_still_disallows_send_alias(self):
        """Alias admission is explicit and does not weaken the old packing contract."""
        with self.assertRaisesRegex(ValueError, "must not alias"):
            self.domain.prepare_accumulated_payload(
                self.active, 2, self._targets(), self.transfer, object(), outputs=self.outputs,
                staging=self.staging)

    def test_no_migration_maps_only_home_views(self):
        """No replica contribution requires no send storage or speculative dW."""
        buffers = (torch.empty(0), torch.empty(0))
        targets = self.domain.accumulation_targets((1, 0), 2, self.outputs, None, staging=buffers)
        for role, output in enumerate(self.outputs):
            self.assertEqual(targets[role][0].data_ptr(), output[1].data_ptr())
            self.assertEqual(targets[role][1].data_ptr(), output[0].data_ptr())
