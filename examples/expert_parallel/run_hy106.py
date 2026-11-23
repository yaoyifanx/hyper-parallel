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
"""Launch VeOmni HY106 with the HP execution-order prefetch adapter.

Use this file as the torchrun worker, passing the normal VeOmni --config flag.
The caller activates VeOmni/CANN and selects the MoonEP backend and optimization
flags. No VeOmni source files or training mathematics are modified.
"""
from collections.abc import Mapping, Sequence
import runpy

from torch import nn

from hyper_parallel.core.fully_shard.prefetch import build_execution_prefetch_plan


def build_hy106_prefetch_plan(model: nn.Module, order: Sequence[nn.Module],
                              groups: Mapping[nn.Module, Sequence[nn.Module]], **kwargs: object) -> dict:
    """Declare HY's router-before-experts execution order and demand-load other subgraphs.

    The model's block order excludes embedding and independent frozen visual
    stacks. They remain fully sharded and demand-loaded. Unknown extra children
    require an explicit adapter update instead of guessing registration order.
    """
    ordered = dict(groups)
    for block in order:
        units = tuple(groups[block])
        moe = getattr(block, "mlp", None)
        if moe is not None and hasattr(moe, "experts") and hasattr(moe, "gate"):
            actual = [block, *[unit for unit in (moe.gate, moe.experts) if unit in units]]
            if set(actual) != set(units):
                raise ValueError("HY block has extra FSDP children; declare their actual execution order")
            ordered[block] = actual
    return build_execution_prefetch_plan(model, order, ordered, **kwargs)


def main() -> None:
    """Install the optional trainer adapter before its FSDP wrapping runs."""
    # VeOmni is an optional application dependency, not a HyperParallel import requirement.
    import veomni.distributed.fsdp2.prefetch as prefetch  # pylint: disable=import-outside-toplevel

    prefetch.build_execution_prefetch_plan = build_hy106_prefetch_plan
    runpy.run_module("tasks.transfusion.train_hy", run_name="__main__")


if __name__ == "__main__":
    main()
