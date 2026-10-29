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
"""Compose Encoder Parallel planning with ordinary forward-input providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyper_parallel.data.batching.runtime_input import RuntimeInputAdapter, RuntimeInputContext
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelPlan, EncoderParallelRuntime


class EncoderParallelInputAdapter(RuntimeInputAdapter):
    """Add one microbatch plan after local validation and other runtime inputs."""

    def __init__(self, *, runtime: EncoderParallelRuntime,
                 inner: RuntimeInputAdapter | None) -> None:
        """Combine visual planning with the batch's existing input provider."""
        if inner is not None and "encoder_parallel_plan" in inner.runtime_input_fields():
            raise ValueError("encoder_parallel_plan has two owners")
        self.runtime = runtime
        self.inner = inner

    def runtime_input_fields(self) -> tuple[str, ...]:
        """Declare the existing fields and the plan field as one input contract."""
        fields = () if self.inner is None else self.inner.runtime_input_fields()
        return (*fields, "encoder_parallel_plan")

    def build_runtime_inputs(
            self, *, batch: Mapping[str, Any], context: RuntimeInputContext,
    ) -> Mapping[str, Any]:
        """Validate locally, build other inputs, then agree on the visual plan."""
        if getattr(getattr(self.runtime, "config", None), "group_scope", "sample") == "global":
            prepared = None
            error = None
            try:
                prepared = self.runtime.preflight_batch(batch)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = exc
            self.runtime.agree_or_raise("preflight", error)
            generated = None
            error = None
            try:
                generated = {} if self.inner is None else self.inner.build_runtime_inputs(batch=batch, context=context)
                if not isinstance(generated, Mapping):
                    raise TypeError("RuntimeInputAdapter.build_runtime_inputs must return a mapping")
                if "encoder_parallel_plan" in batch or "encoder_parallel_plan" in generated \
                        or set(batch).intersection(generated):
                    raise ValueError("runtime inputs cannot replace Omni model inputs")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                error = exc
            self.runtime.agree_or_raise("inner adapter", error)
        else:
            prepared = self.runtime.preflight_batch(batch)
            generated = {} if self.inner is None else self.inner.build_runtime_inputs(batch=batch, context=context)
        if not isinstance(generated, Mapping):
            raise TypeError("RuntimeInputAdapter.build_runtime_inputs must return a mapping")
        inputs = dict(generated)
        if "encoder_parallel_plan" in batch or "encoder_parallel_plan" in inputs or set(batch).intersection(inputs):
            raise ValueError("runtime inputs cannot replace Omni model inputs")
        plan = self.runtime.finalize_plan(*prepared)
        if isinstance(plan, EncoderParallelPlan):
            plan.prepare_patches(batch)
        inputs["encoder_parallel_plan"] = plan
        return inputs
