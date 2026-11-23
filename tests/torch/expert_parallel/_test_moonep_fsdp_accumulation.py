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
"""Four-NPU EP2/FSDP2 no-sync accumulation and replica-owner acceptance."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.multicore.modules.routed_experts import transport
from hyper_parallel.distributed.expert_parallel.moonep import configure_moonep, wrap_moonep_checkpoint
from hyper_parallel.distributed.expert_parallel.moonep.runtime import MoonEPExecutor, local_tensor
from hyper_parallel.distributed.expert_parallel.moonep.fused.executor import SparseMoonEPExecutor


class Experts(torch.nn.Module):
    """Use the actual GLM transposed packed-parameter layout."""

    def __init__(self, gate: torch.Tensor, down: torch.Tensor, mesh: object, backend: str, top_k: int, dedup: bool = False) -> None:
        """Bind EP experts to the physical packed parameter layout."""
        super().__init__()
        self.gate_up = torch.nn.Parameter(gate.clone())
        self.down = torch.nn.Parameter(down.clone())
        constructor = MoonEPExecutor if backend == "nonfused" else SparseMoonEPExecutor
        options = {"activation": "swiglu"} if backend == "nonfused" else {"swiglu_limit": None}
        if dedup:
            options.update(token_dedup=True, dedup_protocol="peer_token_v2", backward_ratio=1.0)
        self._moonep_executor = constructor(self, self.weights, ep_group=mesh.get_group(), ep_size=2,
                                            num_experts=6, top_k=top_k, gradient_accumulation_fusion=True, **options)

    def weights(self) -> tuple:
        """Expose GMM views while keeping physical parameters contiguous."""
        return local_tensor(self.gate_up).transpose(1, 2), local_tensor(self.down).transpose(1, 2)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Use the normal FSDP expert module boundary."""
        return self._moonep_executor(x, ids, scores, expert_weights=self.weights())


class Block(torch.nn.Module):
    """Use the documented checkpoint region around an independently sharded expert."""

    def __init__(self, experts: Experts) -> None:
        """Keep the checkpoint boundary outside the sharded expert."""
        super().__init__()
        self.experts = experts

    def forward(self, x: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Enter expert hooks inside the checkpoint invocation context."""
        return self.experts(x, ids, scores)


@torch.no_grad()
def reference(x: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor, dy: torch.Tensor,
              gate: torch.Tensor, down: torch.Tensor) -> tuple:
    """Compute independent per-expert BF16 activations and FP64 dW products."""
    gg = torch.zeros_like(gate, device="cpu", dtype=torch.float64)
    gd = torch.zeros_like(down, device="cpu", dtype=torch.float64)
    dx = torch.zeros_like(x, dtype=torch.float32)
    ds = torch.zeros_like(scores, device="cpu", dtype=torch.float64)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(gate.shape[0]):
        positions = (ids == expert).nonzero()
        rows, choices = positions[:, 0], positions[:, 1]
        xx = x.index_select(0, rows)
        derivative = (dy.index_select(0, rows).float() * scores[rows, choices, None]).to(x.dtype)
        z = xx @ gate[expert].T
        activation = torch_npu.npu_swiglu(z, dim=-1)
        projected = activation @ down[expert].T
        output.index_add_(0, rows, projected.float() * scores[rows, choices, None])
        ds[rows.cpu(), choices.cpu()] = (projected.cpu().double() * dy.index_select(0, rows).cpu().double()).sum(-1)
        mid = derivative @ down[expert]
        dz = torch_npu.npu_swiglu_backward(mid.contiguous(), z, dim=-1)
        dx.index_add_(0, rows, (dz @ gate[expert]).float())
        gg[expert] = dz.cpu().double().T @ xx.cpu().double()
        gd[expert] = derivative.cpu().double().T @ activation.cpu().double()
    return gg, gd, dx.to(x.dtype), ds, output.to(x.dtype)


def run(args: argparse.Namespace) -> None:
    """Run four short/long/short/long microsteps with migration and delayed FSDP sync."""
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = False
    device = torch.device("npu", torch.npu.current_device())
    dist.init_process_group("hccl")
    control = dist.new_group(backend="gloo")
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("efs", "ep"))
    ep, efs = mesh["ep"], mesh["efs"]
    ep_rank, efs_rank = ep.get_local_rank(), efs.get_local_rank()
    if args.native_dir:
        libraries = tuple(args.native_dir / f"libhyper_parallel_routed_experts_{name}.so"
                          for name in ("kernels", "torch"))
        transport.get_routed_experts_paths = lambda: libraries
    generator = torch.Generator().manual_seed(1729)
    hidden, inner = (4096, 1408) if args.full_shape else (128, 128)
    gate = (torch.randn(6, 2 * inner, hidden, generator=generator) * 0.025).to(device)
    down = (torch.randn(6, hidden, inner, generator=generator) * 0.025).to(device)
    experts = Experts(gate[ep_rank * 3:(ep_rank + 1) * 3], down[ep_rank * 3:(ep_rank + 1) * 3],
                      ep, args.backend, args.top_k, args.dedup)
    block = Block(experts)
    model = wrap_moonep_checkpoint(block) if args.recompute else block
    options = {"mesh": efs, "reshard_after_forward": True, "shard_placement_fn": lambda _: Shard(1),
               "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                  cast_forward_inputs=False)}
    fully_shard(experts, **options)
    fully_shard(model, **options)
    runtime = configure_moonep(model)
    ref_gate = torch.zeros_like(gate, device="cpu", dtype=torch.float64)
    ref_down = torch.zeros_like(down, device="cpu", dtype=torch.float64)
    dx_errors, score_errors, forward_errors, memory, peaks = [], [], [], [], []
    lengths = (32, 8192, 20, 8192) if args.balanced_route else (31, 8193, 19, 8193)
    if args.full_shape:
        lengths = (3, 19, 5, 23)
    actual_lengths = []
    for microstep, tokens in enumerate(lengths):
        if args.empty_source_rank and rank % 2 == 0:
            tokens = 0
        actual_lengths.append(tokens)
        model.set_requires_gradient_sync(microstep == len(lengths) - 1)
        generator.manual_seed(8000 + rank * 31 + microstep)
        x = (torch.randn(tokens, hidden, generator=generator) * 0.1).bfloat16().to(device).requires_grad_()
        ids = torch.zeros((tokens, args.top_k), dtype=torch.int64, device=device)
        if args.balanced_route:
            if args.top_k == 1:
                ids[1::2, 0] = 3
            else:
                ids[:, 1] = 3
        elif args.top_k == 1 and tokens:
            ids[-3:, 0] = torch.tensor([1, 4, 5], device=device)
        elif args.top_k == 2:
            ids[:, 1] = torch.tensor([1, 4, 5], device=device)[torch.arange(tokens, device=device) % 3]
        scores = torch.rand((tokens, args.top_k), generator=generator)
        scores = (scores / scores.sum(-1, keepdim=True)).to(device).requires_grad_()
        dy = (torch.randn(tokens, hidden, generator=generator) * 0.1).bfloat16().to(device)
        expected_g, expected_d, expected_dx, expected_ds, expected_output = reference(
            x.detach(), ids, scores.detach(), dy, gate.bfloat16(), down.bfloat16())
        ref_gate += expected_g
        ref_down += expected_d
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        output = model(x, ids, scores)
        forward_error = float((output.float() - expected_output.float()).norm()
                              / expected_output.float().norm().clamp_min(1e-20))
        if forward_error > 0.015:
            raise AssertionError(f"forward relative error {forward_error}")
        forward_errors.append(forward_error)
        loss = (output.float() * dy.float()).sum()
        if args.backend == "megakernel":
            runtime.backward(loss)
        else:
            loss.backward()
        dx_error = float((x.grad.float() - expected_dx.float()).norm() / expected_dx.float().norm().clamp_min(1e-20))
        if dx_error > 0.015:
            raise AssertionError(f"distributed dX relative error {dx_error}")
        dx_errors.append(dx_error)
        score_error = float((scores.grad.cpu().double() - expected_ds).norm() / expected_ds.norm().clamp_min(1e-30))
        if score_error > 0.015:
            raise AssertionError(f"router score gradient relative error {score_error}")
        score_errors.append(score_error)
        peaks.append({"rows": tokens, "allocated": torch.npu.max_memory_allocated(),
                      "reserved": torch.npu.max_memory_reserved(),
                      "allocator_retries": torch.npu.memory_stats().get("num_alloc_retries", 0)})
        arena = getattr(runtime, "arena", None)
        if arena is not None:
            memory.append(arena.memory_snapshot())
        if microstep < len(lengths) - 1:
            for parameter in experts.hsdp_scheduler.hsdp_state.hsdp_params:
                value = parameter.unsharded_accumulated_grad_data
                if value is None or value.dtype != torch.float32:
                    raise AssertionError("no-sync must retain the real FP32 accumulator")
    # Every destination is replaced by all_gather_object before it is read.
    gathered = [(ref_gate, ref_down) for _ in range(4)]
    dist.all_gather_object(gathered, (ref_gate, ref_down), group=control)
    expected = [sum(item[role] for item in gathered) / 2 for role in range(2)]
    errors = []
    per_expert_errors = []
    gradients = []
    for value, oracle in zip((experts.gate_up, experts.down), expected):
        target = oracle[ep_rank * 3:(ep_rank + 1) * 3].chunk(2, dim=1)[efs_rank]
        actual = local_tensor(value.grad).cpu().double()
        error = float((actual - target).norm() / target.norm().clamp_min(1e-30))
        expert_errors = [(actual[index] - target[index]).norm() / target[index].norm().clamp_min(1e-30)
                         for index in range(actual.shape[0])]
        if error > 0.015 or max(expert_errors) > 0.03:
            raise AssertionError(f"distributed owner dW relative error {error}, per-expert {expert_errors}")
        errors.append(error)
        per_expert_errors.append([float(value) for value in expert_errors])
        if args.save_gradients:
            gradients.append(local_tensor(value.grad).detach().cpu())
    runtime.finish_step()
    if args.balanced_route and runtime.statistics.get("remote_experts_max", 0):
        raise AssertionError("balanced home routing must exercise the no-migration path")
    args.output.mkdir(exist_ok=True, parents=True)
    if args.save_gradients:
        torch.save(gradients, args.output / f"gradients_rank_{rank}.pt")
    (args.output / f"rank_{rank}.json").write_text(json.dumps({"backend": args.backend, "recompute": args.recompute,
        "dw_relative_l2": errors, "per_expert_dw_relative_l2": per_expert_errors,
        "dx_relative_l2": dx_errors, "microstep_rows": actual_lengths,
        "empty_source_rank": args.empty_source_rank,
        "hidden": hidden, "intermediate": inner,
        "forward_relative_l2": forward_errors, "score_relative_l2": score_errors,
        "top_k": args.top_k, "balanced_route": args.balanced_route,
        "arena": memory, "memory_peaks": peaks, "statistics": dict(runtime.statistics)}, indent=2) + "\n")
    if args.backend == "megakernel":
        runtime.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nonfused", "megakernel"), required=True)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--dedup", action="store_true")
    parser.add_argument("--top-k", type=int, choices=(1, 2), default=1)
    parser.add_argument("--balanced-route", action="store_true")
    parser.add_argument("--empty-source-rank", action="store_true")
    parser.add_argument("--full-shape", action="store_true")
    parser.add_argument("--save-gradients", action="store_true")
    run(parser.parse_args())
