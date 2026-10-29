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
"""Unit tests for VLM micro-batch lifetime."""

from __future__ import annotations

import unittest
import weakref
from collections.abc import Callable, Iterator, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch

from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from hyper_parallel.trainer.vlm_trainer import VLMTrainer
from tests.common.mark_utils import arg_mark


class TestVLMMicroBatchLifetime(unittest.TestCase):
    """Keep device batches live only during their own forward/backward."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_reads_next_batch_after_previous_backward_and_release(self):
        """Verify streaming preserves step accounting and frees old inputs.

        Feature: VLM gradient accumulation.
        Description: Read three batches whose tensors are watched by weak references.
        Expectation: Each read follows the previous backward and release.
        """
        events = []
        batch_refs = []
        token_counts = []

        class BatchReader:
            """Model a CPU source queue with one live device batch."""

            @staticmethod
            def prepare_step_sources(
                    iterator: Iterator[dict[str, torch.Tensor]], num_micro_steps: int,
                    count_tokens: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
            ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
                """Prefetch source batches and count actual target tokens."""
                sources = [next(iterator) for _ in range(num_micro_steps)]
                counts = [count_tokens(source)["foundation_tokens"] for source in sources]
                events.append("prefetch")
                return sources, {"foundation_tokens": sum(counts)}

            def __call__(
                    self, _iterator: Any, *, prepared_cp_batch: Mapping[str, torch.Tensor] | None = None,
            ) -> tuple[dict[str, torch.Tensor], Mapping[str, torch.Tensor]]:
                """Create one watched device batch from a prepared CPU batch."""
                if batch_refs:
                    self_test.assertIsNone(batch_refs[-1](), "previous device batch is still live")
                    self_test.assertEqual(events[-1], "backward")
                value = torch.ones(8)
                batch_refs.append(weakref.ref(value))
                events.append("get_batch")
                return {"input_ids": value}, prepared_cp_batch

        self_test = self

        def forward_backward_step(
                _model_inputs: Mapping[str, torch.Tensor], _loss_inputs: Mapping[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            """Record the denominator observed by each backward call."""
            token_counts.append(int(trainer.base.step_token_counts["foundation_tokens"]))
            events.append("backward")
            return torch.tensor(1.0), {"language": torch.tensor(0.5)}

        trainer = VLMTrainer.__new__(VLMTrainer)
        trainer.base = SimpleNamespace(
            num_micro_batches=3,
            get_batch=BatchReader(),
            on_step_begin=lambda **_kwargs: events.append("step_begin"),
            on_micro_step_begin=lambda _batch: events.append("micro_begin"),
            model_integration=SimpleNamespace(begin_step=Mock()),
            state=SimpleNamespace(global_step=0),
            model_reshard=Mock(),
            configure_fsdp_gradient_sync=Mock(),
            begin_fsdp_runtime_diagnostics=Mock(),
            forward_backward_step=forward_backward_step,
            prepare_optimizer_step=lambda: 0.5,
            step_optimizers_and_schedulers=Mock(),
            _end_model_integration_step=Mock(),
            on_step_end=Mock(),
        )
        with (patch("hyper_parallel.trainer.vlm_trainer.memory_profiler.step"),
              patch("hyper_parallel.trainer.vlm_trainer.synchronize")):
            metrics = trainer.train_step(iter({
                "labels": torch.zeros(1, count + 1, dtype=torch.long),
            } for count in (2, 4, 7)))

        self.assertEqual(events, ["step_begin", "prefetch", "get_batch", "micro_begin", "backward",
                                  "get_batch", "micro_begin", "backward",
                                  "get_batch", "micro_begin", "backward"])
        self.assertEqual(token_counts, [13, 13, 13])
        self.assertTrue(all(reference() is None for reference in batch_refs))
        self.assertEqual(metrics, {"loss": 3.0, "grad_norm": 0.5})

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_omni_counts_cp_local_tokens_before_device_transport(self):
        """Count real source tokens while leaving the batches on CPU.

        Feature: Omni micro-batch source prefetch.
        Description: Count three different CP-local loss masks.
        Expectation: The exact total is returned and no tensor moves to device.
        """
        reader = OmniParallelBatch.__new__(OmniParallelBatch)
        reader.parallel_context = SimpleNamespace(build_on_rank=lambda: True, tp_world_size=1,
                                                  distributed_enabled=False)
        reader.cp_sharder = SimpleNamespace(shard=lambda batch: batch)
        reader.device = torch.device("cpu")
        reader.source_only_pixel_values = False
        sources = ({
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "labels": torch.zeros(1, 5, dtype=torch.long),
            "loss_mask": torch.tensor([[0, 1, 1, 0, flag]], dtype=torch.bool),
        } for flag in (0, 1, 0))

        batches, counts = reader.prepare_step_sources(iter(sources), 3, count_loss_token)

        self.assertEqual(len(batches), 3)
        self.assertEqual(int(counts["foundation_tokens"]), 7)
        self.assertTrue(all(batch["labels"].device.type == "cpu" for batch in batches))
