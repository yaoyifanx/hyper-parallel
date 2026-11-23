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
"""FP32 accumulation acceptance across microsteps and empty experts."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hyper_parallel.distributed.expert_parallel.moonep.fsdp_hooks import MoonEPGradientExtension
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_utils import FSDPSchedulerState
from hyper_parallel.distributed.expert_parallel.moonep.fused.weights import TransferDomain
from threading import RLock

import torch

from hyper_parallel.distributed.expert_parallel.moonep.numerics import grouped_weight_grad_add


class TestWgradAccumulation(unittest.TestCase):
    """Use an independent double-precision per-expert reference."""

    def test_microsteps_and_transposed_parameter(self):
        """Home contributions must accumulate without an intervening BF16 round."""
        generator = torch.Generator().manual_seed(2718)
        for transpose in (False, True):
            shape = (3, 5, 4) if transpose else (3, 4, 5)
            target = torch.randn(shape, generator=generator)
            reference = target.double().clone()
            for counts in ([2, 0, 4], [0, 3, 0], [0, 0, 0]):
                x = torch.randn(sum(counts), 4, generator=generator).bfloat16()
                dy = torch.randn(sum(counts), 5, generator=generator).bfloat16()
                boundaries = torch.tensor(counts).cumsum(0)
                grouped_weight_grad_add(target, x, dy, boundaries, transposed=transpose)
                start = 0
                for index, count in enumerate(counts):
                    value = x[start:start + count].double().T @ dy[start:start + count].double()
                    reference[index] += value.T if transpose else value
                    start += count
            torch.testing.assert_close(target.double(), reference, rtol=1e-6, atol=1e-6)


class TestFSDPAccumulationOwnership(unittest.TestCase):
    """Remote BF16 values must be added into the real FP32 owner gradient."""

    def test_repeated_peer_contributions_preserve_fp32_sum(self):
        """Never merge a remote pair in BF16 before updating the owner g."""
        targets = [torch.full((2, 3, 4), 1024.0), torch.full((2, 5, 3), 1024.0)]
        parameters = [SimpleNamespace(unsharded_accumulated_grad_data=value) for value in targets]
        packet = SimpleNamespace(receive_buffer=torch.empty(0),
                                 plan=SimpleNamespace(send_slots=(1, 1), send_ids=torch.tensor([1, 1])))
        remote = (torch.full((2, 3, 4), 0.00390625, dtype=torch.bfloat16),
                  torch.full((2, 3, 5), 0.00390625, dtype=torch.bfloat16))
        runtime = SimpleNamespace(wait_gradient_return=lambda _: remote, statistics={"gradient_waits": 0})
        extension = MoonEPGradientExtension(SimpleNamespace(), runtime)
        extension.pending = [packet]
        with patch.object(extension, "_layouts", return_value=((parameters[0], False), (parameters[1], True))), \
                patch.object(extension, "uses_accumulated_grad", return_value=True):
            extension.complete(SimpleNamespace())
        for target in targets:
            torch.testing.assert_close(target[0], torch.full_like(target[0], 1024.0), rtol=0, atol=0)
            torch.testing.assert_close(target[1], torch.full_like(target[1], 1024.0078125), rtol=0, atol=0)
        self.assertEqual(extension.pending, [])

    def test_notify_none_autograd_gradients(self):
        """Explicit readiness replaces hooks when custom backward returns None."""
        tensors = (torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3)))
        parameters = [SimpleNamespace(unsharded_param=value) for value in tensors]
        scheduler = SimpleNamespace(hsdp_state=object(), _on_param_grad_ready=Mock())
        extension = MoonEPGradientExtension(SimpleNamespace(), SimpleNamespace())
        with patch.object(extension, "_layouts", return_value=((parameters[0], False), (parameters[1], False))):
            extension.notify_accumulated(scheduler)
        self.assertEqual(scheduler._on_param_grad_ready.call_count, 2)
        self.assertTrue(all(value.grad is None for value in tensors))


    def test_veomni_marker_honors_accumulation_switch(self):
        """Automatic integration activates only with the actual model installation marker."""
        parameters = [SimpleNamespace(unsharded_param=torch.nn.Parameter(torch.ones(2))) for _ in range(2)]
        executor = SimpleNamespace(gradient_accumulation_fusion=None)
        extension = MoonEPGradientExtension(executor, SimpleNamespace())
        layouts = tuple((parameter, False) for parameter in parameters)
        with patch.dict("os.environ", {"HP_MOONEP_FUSED_WGRAD": "1"}), patch.object(extension, "_layouts", return_value=layouts):
            self.assertFalse(extension.uses_accumulated_grad(SimpleNamespace()))
            for parameter in parameters:
                parameter.unsharded_param._veomni_fused_wgrad_owner = parameter
            with patch.dict("os.environ", {"HP_MOONEP_FUSED_WGRAD": "0"}):
                self.assertTrue(extension.uses_accumulated_grad(SimpleNamespace()))
            executor.gradient_accumulation_fusion = False
            self.assertFalse(extension.uses_accumulated_grad(SimpleNamespace()))


class TestExpertBoundaryReplay(unittest.TestCase):
    """Replay at a pre-restored expert boundary must still prepare its invocation."""

    def test_replay_notifies_inputs_without_resubmitting_ag(self):
        """Notify pre-forward then completion while preserving the backward scheduler state."""
        events = []
        extension = SimpleNamespace(fsdp_pre_forward=lambda *_: events.append("inputs"))
        state = SimpleNamespace(all_gather_extension=extension,
                                complete_unshard=lambda: events.append("complete"), unshard=Mock())
        scheduler = SimpleNamespace(scheduler_state=FSDPSchedulerState.PRE_BACKWARD, hsdp_state=state)
        args, kwargs = (torch.ones(2),), {}
        result = HSDPSchedulerV2._hsdp_forward_pre_hook(scheduler, None, args, kwargs)
        self.assertEqual(events, ["inputs", "complete"])
        self.assertIs(result[0], args)
        state.unshard.assert_not_called()
        self.assertEqual(scheduler.scheduler_state, FSDPSchedulerState.PRE_BACKWARD)


class TestEmptyGradientExchange(unittest.TestCase):
    """An empty G exchange must not retain home g on an unrelated stream."""

    def test_no_migration_uses_only_current_compute_fence(self):
        """Skip producer polling and the communication stream when no peer contribution exists."""
        domain = TransferDomain.__new__(TransferDomain)
        domain._lock, domain.poisoned = RLock(), False
        domain.device = torch.device("cpu")
        packet = SimpleNamespace(owner=domain, ready=None, finished=False, transfer=None, producer=Mock())
        event, stream = Mock(), object()
        module = SimpleNamespace(Event=lambda: event, current_stream=lambda _device: stream)
        with patch("hyper_parallel.distributed.expert_parallel.moonep.fused.weights.torch.npu", module,
                   create=True):
            domain.submit_gradient_payload(packet)
        event.record.assert_called_once_with(stream)
        packet.producer.wait_on_stream.assert_not_called()
        self.assertFalse(domain.poisoned)
