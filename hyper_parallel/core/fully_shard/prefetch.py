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
"""Build bounded prefetch plans for an explicitly ordered model subgraph."""
from collections.abc import Mapping, Sequence

from torch import nn


def build_execution_prefetch_plan(model: nn.Module, execution_order_blocks: Sequence[nn.Module],
                                  fsdp_unit_groups: Mapping[nn.Module, Sequence[nn.Module]], *,
                                  forward_depth: int, backward_depth: int, recompute_forward: bool) -> dict:
    """Describe prefetch targets without changing backend state.

    Args:
        model: Root used to resolve unit FQNs.
        execution_order_blocks: Explicit forward order of the selected block subgraph.
        fsdp_unit_groups: Each group starts with its parent block, followed by its
            independently sharded children in actual forward execution order.
            Extra groups outside the declared subgraph remain demand-loaded and
            are listed in the result; registration order is never inferred.
        forward_depth: Number of future flattened units, not decoder layers.
        backward_depth: Number of future flattened units in backward order.
        recompute_forward: Attach forward targets only to block boundaries.

    Returns:
        A serializable plan, including excluded demand-loaded unit names.

    Raises:
        ValueError: Invalid depths, missing blocks, duplicate units or unnamed modules.
    """
    for depth in (forward_depth, backward_depth):
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("prefetch depths must be nonnegative integers")
    blocks = tuple(execution_order_blocks)
    if len(set(blocks)) != len(blocks):
        raise ValueError("execution order contains a duplicate block")
    if any(block not in fsdp_unit_groups for block in blocks):
        raise ValueError("each declared execution block must have an explicit FSDP unit group")
    names = {module: name for name, module in model.named_modules()}
    groups = []
    for block in blocks:
        units = tuple(fsdp_unit_groups[block])
        if not units or units[0] is not block or len(set(units)) != len(units):
            raise ValueError("each unit group must start with its block and contain distinct ordered units")
        if any(unit not in names for unit in units):
            raise ValueError("prefetch units must be named modules belonging to the root model")
        groups.append(units)
    forward = tuple(unit for units in groups for unit in units)
    if len(set(forward)) != len(forward):
        raise ValueError("a prefetch unit cannot appear in multiple block groups")
    backward = tuple(unit for units in reversed(groups) for unit in (units[0], *reversed(units[1:])))
    targets = {names[unit]: {"forward": [], "backward": []} for unit in forward}
    if recompute_forward:
        for index, units in enumerate(groups):
            future = tuple(unit for following in groups[index + 1:] for unit in following)[:forward_depth]
            targets[names[units[0]]]["forward"] = [names[unit] for unit in future]
    else:
        for index, unit in enumerate(forward):
            targets[names[unit]]["forward"] = [names[future] for future in forward[index + 1:index + 1 + forward_depth]]
    for index, unit in enumerate(backward):
        targets[names[unit]]["backward"] = [names[future] for future in backward[index + 1:index + 1 + backward_depth]]
    excluded = sorted({names[unit] for block, units in fsdp_unit_groups.items() if block not in blocks
                       for unit in units if unit in names and unit not in forward})
    counts = {direction: [len(value[direction]) for value in targets.values()] for direction in ("forward", "backward")}
    return {"strategy": "execution", "block_order": [names[block] for block in blocks],
            "forward_depth": forward_depth, "backward_depth": backward_depth,
            "recompute_forward": recompute_forward, "units": targets, "demand_loaded_units": excluded,
            "inflight": {"n_blocks": len(blocks), "n_units": len(forward),
                         "fwd_max": max(counts["forward"], default=0), "fwd_total": sum(counts["forward"]),
                         "bwd_max": max(counts["backward"], default=0), "bwd_total": sum(counts["backward"])}}


def apply_execution_prefetch_plan(model: nn.Module, plan: dict) -> None:
    """Apply explicit targets through the public FSDP module setters."""
    modules = dict(model.named_modules())
    for name, targets in plan["units"].items():
        unit = modules[name]
        unit.set_modules_to_forward_prefetch([modules[target] for target in targets["forward"]])
        unit.set_modules_to_backward_prefetch([modules[target] for target in targets["backward"]])
