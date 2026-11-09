# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""expert_parallel: the Qwen3-MoE EP compute archetype factory.

Migrated from ``distributed/expert_parallel/recipes.py`` in M3 (adjust doc
§5.1/§5.4): only the Qwen-specific router contract (``TopKRouter`` module
returning ``(logits, scores, indices)``, no shared expert) sinks into the
model adapter. The combine skeleton, the router-adapter registry and the
dispatch/collective primitives stay in the generic
``auto_models/distributed/expert_parallel`` layer; this factory composes
them through the now-public ``build_ep_compute``.
"""

from typing import Any, Callable

import torch

from hyper_parallel.core.dtensor.dtensor import DTensor

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.distributed._builder.forward_rewriter import _install_bound_forward

from hyper_parallel.distributed.expert_parallel.experts import (
    SHARED_MERGE_MODES,
)
from hyper_parallel.distributed.expert_parallel.recipes import (
    build_ep_compute,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.recipe_spec import (
    local_compute,
)


@local_compute
def qwen3moe_ep_compute_fn(
    *,
    module: Any,
    mesh: Any,
    tp_mesh: Any,
    cp_mesh: Any,
    ep_mesh: Any,
    use_grouped_gemm: bool = False,
    fix_router: bool = False,
    use_megamoe: bool = False,
) -> Callable:
    """Bind ordinary EP or standalone MegaMoE using the original expert parameters.

    Args:
        module: Model block owning the router and expert Parameters.
        mesh: Model mesh supplied by the adapter infrastructure.
        tp_mesh: Tensor-parallel mesh managed by the outer layout.
        cp_mesh: Context-parallel mesh managed by the outer layout.
        ep_mesh: Expert communication mesh.
        use_grouped_gemm: Select grouped GEMM in ordinary EP.
        fix_router: Use balanced destinations only for ordinary EP benchmarks.
        use_megamoe: Select the standalone native MegaMoE executor.

    Returns:
        The local compute callable for the selected expert path.
    """
    del mesh, tp_mesh, cp_mesh
    if use_megamoe:
        if fix_router:
            raise ValueError("Qwen MegaMoE不支持仅供性能测试的均衡路由")
        if ep_mesh is None:
            raise ValueError("Qwen MegaMoE需要显式EP mesh")
        experts = module.experts
        # Native executor has no parameters; the original expert module remains the FSDP owner.
        object.__setattr__(experts, "_qwen_mega_executor", MegaMoeExperts(
            hidden_size=experts.hidden_size, intermediate_size=experts.intermediate_size,
            num_experts=module.gate.num_experts, top_k=module.gate.top_k,
            ep_group=ep_mesh.get_group("ep"), create_parameters=False, swiglu_limit=None,
        ))
        _install_bound_forward(experts, _qwen_megamoe_experts)
        _install_bound_forward(module.gate, _qwen_megamoe_router)
        return _qwen_megamoe_compute
    return build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["qwen3moe"],
        archetype_key="qwen3moe_topk_router",
        expected_attrs=["gate", "experts"],
        combine=SHARED_MERGE_MODES["none"],
        use_grouped_gemm=use_grouped_gemm,
        fix_router=fix_router,
    )


def _qwen_expert_weights(experts: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """读取真实参数的GMM视图；例如已转置的GroupedExperts直接返回[P,H,2I]。"""
    gate, down = experts.gate_up_proj, experts.down_proj
    gate = gate.to_local() if isinstance(gate, DTensor) else gate
    down = down.to_local() if isinstance(down, DTensor) else down
    if not getattr(experts, "is_transposed", False):
        gate, down = gate.transpose(1, 2), down.transpose(1, 2)
    return gate, down


def _qwen_megamoe_experts(experts: Any, hidden: torch.Tensor, ids: torch.Tensor,
                          scores: torch.Tensor) -> torch.Tensor:
    """保留Qwen专家参数的FSDP归属，使用完整MegaMoE路由和计算。"""
    return experts._qwen_mega_executor(  # pylint: disable=protected-access
        hidden, ids, scores, expert_weights=_qwen_expert_weights(experts),
    )


def _qwen_megamoe_router(router: Any, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve the Qwen router formula and differentiable TopK scores."""
    logits = torch.nn.functional.linear(hidden.reshape(-1, router.hidden_dim), router.weight)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    indices = probabilities.topk(router.top_k, dim=-1).indices
    scores = probabilities.gather(-1, indices)
    if router.norm_topk_prob:
        scores = scores / scores.sum(-1, keepdim=True)
    return logits, scores.to(logits.dtype), indices


def _qwen_megamoe_compute(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """路由后调用实际experts子模块；例如[1,S,H]输入最终恢复为同一形状。"""
    indices, scores = MOE_ROUTER_ADAPTERS["qwen3moe"](module, hidden_states)
    return module.experts(hidden_states, indices, scores)


__all__ = [
    "qwen3moe_ep_compute_fn",
]
