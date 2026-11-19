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
"""An unused speculative gather must not suppress the next iteration's gather."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from hyper_parallel.core.fully_shard.hsdp_param import AllGatherCommCtx
from hyper_parallel.core.fully_shard.hsdp_scheduler import HSDPSchedulerContext, HSDPSchedulerV2
from hyper_parallel.core.fully_shard.hsdp_state import HSDPStateV2
from hyper_parallel.core.fully_shard.param_group import AllGatherResult
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestUnusedPrefetchLifecycle(unittest.TestCase):
    """Exercise actual state transitions for fused and per-parameter gathers."""

    def _case(self, fused: bool, failure: bool) -> None:
        state = object.__new__(HSDPStateV2)
        state.is_shard = True
        state.module_name = "unused"
        state.all_gather_extension = None
        state._all_gather_started = False
        state._prefetch_pending = False
        stale_event = object()
        state._unshard_event = stale_event
        state.comm_fusion_policy = SimpleNamespace(enable_comm_fusion=fused)
        parameter = SimpleNamespace(allgather_comm_ctx=AllGatherCommCtx(), shard_world_size=2,
                                    free_unsharded_param=Mock())
        state.hsdp_params = [parameter]
        handle = Mock()
        bucket = SimpleNamespace(hsdp_params=[parameter], all_gather_result=None)

        def submit(_async_op: bool) -> None:
            if fused:
                bucket.all_gather_result = AllGatherResult(object(), object(), handle)
            else:
                parameter.allgather_comm_ctx = AllGatherCommCtx(
                    allgather_handle=handle, allgather_input=object(), allgather_output=object())

        gather = Mock(side_effect=submit)
        parameter.unshard = gather
        state.param_group = SimpleNamespace(all_gather_buckets=[bucket], unshard=gather) if fused else None
        scheduler = object.__new__(HSDPSchedulerV2)
        scheduler.scheduler_ctx = HSDPSchedulerContext()
        scheduler.scheduler_ctx.all_hsdp_schedulers = [SimpleNamespace(hsdp_state=state)]
        state.prefetch()
        self.assertTrue(state._all_gather_started)
        self.assertTrue(state._prefetch_pending)

        def wait() -> None:
            self.assertTrue(state._all_gather_started)
            parameter.free_unsharded_param.assert_not_called()
            if failure:
                raise RuntimeError("unfinished gather")

        handle.wait.side_effect = wait
        if failure:
            with self.assertRaisesRegex(RuntimeError, "unfinished gather"):
                scheduler._finalize_unused_prefetches()
            self.assertTrue(state._all_gather_started)
            self.assertTrue(state._prefetch_pending)
            self.assertIs(state._unshard_event, stale_event)
            return
        scheduler._finalize_unused_prefetches()
        handle.wait.assert_called_once_with()
        self.assertTrue(state.is_shard)
        self.assertFalse(state._all_gather_started)
        self.assertFalse(state._prefetch_pending)
        self.assertIsNone(state._unshard_event)
        if not fused:
            parameter.free_unsharded_param.assert_called_once_with()
            self.assertIsNone(parameter.allgather_comm_ctx.allgather_input)
        state.unshard(async_op=True)
        self.assertEqual(gather.call_count, 2)

    def test_next_iteration_submits_a_new_gather(self) -> None:
        """Both communication paths relaunch after an unused gather is retired."""
        for fused in (False, True):
            with self.subTest(fused=fused):
                self._case(fused, False)

    def test_failed_wait_preserves_inflight_ownership(self) -> None:
        """A failed gather must retain its state and live storage references."""
        for fused in (False, True):
            with self.subTest(fused=fused):
                self._case(fused, True)
