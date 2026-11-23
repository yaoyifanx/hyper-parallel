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
"""Verify collective submission order across FSDP notification timings on CPU."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fsdp_hooks import MoonEPAllGatherExtension
from hyper_parallel.distributed.expert_parallel.moonep.lifecycle import ExecutionKey, MoonEPLifecycle
from tests.common.mark_utils import arg_mark


class _RecordedLifecycle(MoonEPLifecycle):
    """Record materializations without a transport or device event implementation."""

    def __init__(self, count: int) -> None:
        """Create isolated CPU expert owners.

        Args:
            count: Number of expert calls retained in forward order.
        """
        weight = torch.ones(1)
        executors = [SimpleNamespace(read_weights=lambda: (weight, weight),
                                     owner=SimpleNamespace(forward=lambda *args: None)) for _ in range(count)]
        super().__init__(executors)
        self.fsdp_layers.update(range(count))
        self.keys = [self.new_key(executor) for executor in executors]
        self.executions.update({key: (executor, None) for key, executor in zip(self.keys, executors)})
        self.submissions = []

    def prepare(self, key: ExecutionKey, *, ready_events: tuple = ()) -> Any:
        """Record a producer submission and its device dependencies.

        Args:
            key: Invocation being materialized.
            ready_events: Producer fences attached to the submission.

        Returns:
            An opaque lease consumed by the same execution.
        """
        self.submissions.append((key, ready_events))
        self.weight_leases[key] = object()
        return self.weight_leases[key]

    def retire_backward(self, key: ExecutionKey) -> None:
        """Model release and completion before an expert exit.

        Args:
            key: Current reverse-order invocation.
        """
        self.weight_leases.pop(key, None)
        self.complete(key)
        self.prefetch_after_backward(key, torch.device("cpu"))


class TestRestoredWeightOrder(unittest.TestCase):
    """Exercise early, late, absent and deep prefetch notifications."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_early_home_is_deferred_until_current_ep_exit(self) -> None:
        """Feature: Early restoration.

        Description: Restore a future expert before the current backward exits.
        Expectation: Submit only after the current expert completes.
        """
        runtime = _RecordedLifecycle(2)
        future, current = runtime.keys
        with patch.object(runtime, "_record_weight_dependency", side_effect=["home", "ep_exit"]):
            runtime.prepare_restored_weights(future)
            self.assertEqual(runtime.submissions, [])
            runtime.retire_backward(current)
        self.assertEqual(runtime.submissions, [(future, ("home", "ep_exit"))])
        self.assertEqual(runtime.pending_weight_preparations, {})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_late_home_uses_exit_recorded_without_pending_work(self) -> None:
        """Feature: Late restoration.

        Description: Record an expert exit before the next home notification.
        Expectation: The late submission carries both home and exit fences.
        """
        runtime = _RecordedLifecycle(2)
        future, current = runtime.keys
        with patch.object(runtime, "_record_weight_dependency", side_effect=["ep_exit", "late_home"]):
            runtime.retire_backward(current)
            self.assertEqual(runtime.submissions, [])
            runtime.prepare_restored_weights(future)
        self.assertEqual(runtime.submissions, [(future, ("late_home", "ep_exit"))])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_depth_zero_restores_each_demand_without_waiting_for_itself(self) -> None:
        """Feature: Demand restoration.

        Description: Restore each current invocation without earlier prefetch.
        Expectation: All calls complete in reverse order with no residual state.
        """
        runtime = _RecordedLifecycle(4)
        for key in reversed(runtime.keys):
            runtime.prepare_restored_weights(key)
            self.assertIsNotNone(runtime.require_weight_lease(key))
            runtime.retire_backward(key)
        self.assertEqual([key for key, _ in runtime.submissions], list(reversed(runtime.keys)))
        self.assertIsNone(runtime.backward_weight_ready)
        self.assertEqual(runtime.pending_weight_preparations, {})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_deep_prefetch_releases_only_the_immediate_successor(self) -> None:
        """Feature: Deep prefetch.

        Description: Restore multiple future experts before the current exit.
        Expectation: Only the immediate successor materializes at each exit.
        """
        runtime = _RecordedLifecycle(4)
        for key in runtime.keys[:-1]:
            runtime.prepare_restored_weights(key)
        self.assertEqual(runtime.submissions, [])
        runtime.retire_backward(runtime.keys[-1])
        self.assertEqual([key for key, _ in runtime.submissions], [runtime.keys[-2]])
        self.assertEqual(set(runtime.pending_weight_preparations), set(runtime.keys[:2]))
        runtime.retire_backward(runtime.keys[-2])
        self.assertEqual([key for key, _ in runtime.submissions], list(reversed(runtime.keys[1:3])))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_first_invocation_can_start_immediately(self) -> None:
        """Feature: Backward startup.

        Description: Restore the first current invocation with no predecessor.
        Expectation: Preparation requires only its home restoration fence.
        """
        runtime = _RecordedLifecycle(2)
        current = runtime.keys[-1]
        with patch.object(runtime, "_record_weight_dependency", return_value="home"):
            runtime.prepare_restored_weights(current)
        self.assertEqual(runtime.submissions, [(current, ("home",))])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_repeated_notification_does_not_materialize_twice(self) -> None:
        """Feature: Duplicate restoration.

        Description: Deliver two restoration notifications for one lease.
        Expectation: Exactly one materialization occurs.
        """
        runtime = _RecordedLifecycle(1)
        key = runtime.keys[0]
        runtime.prepare_restored_weights(key)
        runtime.prepare_restored_weights(key)
        self.assertEqual([submitted for submitted, _ in runtime.submissions], [key])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_queued_notification_is_retained_without_polling(self) -> None:
        """Feature: Queued restoration.

        Description: Repeat an early notification before the current exit.
        Expectation: Keep one producer fence and submit once without polling.
        """
        runtime = _RecordedLifecycle(2)
        future, current = runtime.keys
        with patch.object(runtime, "_record_weight_dependency", side_effect=["first_home", "ep_exit"]) as record:
            runtime.prepare_restored_weights(future)
            runtime.prepare_restored_weights(future)
            self.assertEqual(record.call_count, 1)
            runtime.retire_backward(current)
        self.assertEqual(runtime.submissions, [(future, ("first_home", "ep_exit"))])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_fsdp_hook_registers_future_without_running_its_collective(self) -> None:
        """Feature: Restoration extension.

        Description: Notify a future expert through its registered extension.
        Expectation: No collective runs until the current expert exits.
        """
        runtime = _RecordedLifecycle(2)
        future, current = runtime.keys
        runtime.prefetch_key = lambda executor: runtime.keys[executor.layer]
        hook = MoonEPAllGatherExtension(runtime.executors[0], runtime)
        hook.fsdp_post_unshard(None)
        hook.fsdp_post_unshard(None)
        self.assertEqual(runtime.submissions, [])
        self.assertEqual(runtime.statistics["fsdp_ep_prefetches"], 1)
        runtime.retire_backward(current)
        self.assertEqual([key for key, _ in runtime.submissions], [future])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_exit_before_completion_is_rejected(self) -> None:
        """Feature: Exit ordering.

        Description: Attempt to release a successor before current completion.
        Expectation: Reject the invalid lifecycle boundary.
        """
        runtime = _RecordedLifecycle(1)
        with self.assertRaisesRegex(RuntimeError, "completion"):
            runtime.prefetch_after_backward(runtime.keys[0], torch.device("cpu"))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_stale_restoration_is_rejected(self) -> None:
        """Feature: Version isolation.

        Description: Deliver a restoration notification for another version.
        Expectation: Reject the stale execution key.
        """
        runtime = _RecordedLifecycle(1)
        with self.assertRaisesRegex(RuntimeError, "current version"):
            runtime.prepare_restored_weights(replace(runtime.keys[0], version=1))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_missing_demand_lease_still_fails_without_implicit_communication(self) -> None:
        """Feature: Strict consumption.

        Description: Consume a current invocation without parameter restoration.
        Expectation: Fail without implicitly launching communication.
        """
        runtime = _RecordedLifecycle(1)
        with self.assertRaisesRegex(RuntimeError, "restoration hooks"):
            runtime.require_weight_lease(runtime.keys[0])
        self.assertEqual(runtime.submissions, [])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_standalone_does_not_gain_fsdp_prefetch(self) -> None:
        """Feature: Standalone isolation.

        Description: Complete a runtime without restoration extensions.
        Expectation: Do not introduce a new prefetch fence or submission.
        """
        runtime = _RecordedLifecycle(2)
        runtime.fsdp_layers.clear()
        runtime.retire_backward(runtime.keys[-1])
        self.assertIsNone(runtime.backward_weight_ready)
        self.assertEqual(runtime.submissions, [])
