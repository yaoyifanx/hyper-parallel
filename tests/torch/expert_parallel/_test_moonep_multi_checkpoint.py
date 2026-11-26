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
"""Compare a two-expert checkpoint with separate regions on four EP2 x FSDP2 ranks."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from hyper_parallel import fully_shard, init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.core.multicore.modules.routed_experts import transport
from hyper_parallel.distributed.expert_parallel.moonep import configure_moonep, wrap_moonep_checkpoint
from hyper_parallel.distributed.expert_parallel.moonep.runtime import local_tensor
from tests.torch.expert_parallel._test_moonep_fsdp_accumulation import Block, Experts


class TwoStreams(torch.nn.Module):
    """Exercise sibling expert branches whose gradients meet at a shared input."""

    def __init__(self, branches: list) -> None:
        super().__init__()
        self.branches = torch.nn.ModuleList(branches)

    def forward(self, x: torch.Tensor, ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        u = self.branches[0](x.sin(), ids[0], scores[0])
        g = self.branches[1](x.cos(), ids[1], scores[1])
        return u + g


def run(args: argparse.Namespace) -> None:
    """Run separate torchruns for split and whole and compare all local gradients."""
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.config.allow_internal_format = False
    device = torch.device("npu", torch.npu.current_device())
    dist.init_process_group("hccl")
    mesh = init_device_mesh("npu", (2, 2), mesh_dim_names=("efs", "ep"))
    ep, efs = mesh["ep"], mesh["efs"]
    if args.native_dir:
        transport.get_routed_experts_paths = lambda: tuple(
            args.native_dir / f"libhyper_parallel_routed_experts_{name}.so" for name in ("kernels", "torch"))
    generator = torch.Generator().manual_seed(1729)
    experts, branches = [], []
    for _ in range(2):
        gate = torch.randn(6, 256, 128, generator=generator) * 0.025
        down = torch.randn(6, 128, 128, generator=generator) * 0.025
        expert = Experts(gate.chunk(2)[ep.get_local_rank()].to(device),
                         down.chunk(2)[ep.get_local_rank()].to(device), ep, args.backend, 2)
        experts.append(expert)
        branch = Block(expert)
        branches.append(wrap_moonep_checkpoint(branch) if args.layout == "split" else branch)
    model = TwoStreams(branches)
    if args.layout == "whole":
        model = wrap_moonep_checkpoint(model)
    options = dict(mesh=efs, reshard_after_forward=True, shard_placement_fn=lambda _: Shard(1),
                   mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                                  cast_forward_inputs=False))
    for expert in experts:
        fully_shard(expert, **options)
    fully_shard(model, **options)
    experts[0].set_modules_to_forward_prefetch([experts[1]])
    experts[1].set_modules_to_backward_prefetch([experts[0]])
    runtime = configure_moonep(model)
    generator.manual_seed(9000 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    for step, rows in enumerate((64, 95)):
        model.zero_grad(set_to_none=True)
        x = torch.randn(rows, 128, generator=generator).bfloat16().to(device).requires_grad_()
        ids = torch.randint(1, 6, (2, rows, 2), generator=generator)
        ids[:, :, 0] = 0
        scores = torch.rand(2, rows, 2, generator=generator).to(device).requires_grad_()
        dy = torch.randn(rows, 128, generator=generator).to(device)
        y = model(x, ids.to(device), scores)
        loss = (y.float() * dy).sum()
        if args.backend == "megakernel":
            runtime.backward(loss)
        else:
            loss.backward()
        runtime.finish_step()
        assert not runtime.replay_released and not runtime.weight_leases
        assert runtime.statistics["recomputations"] == 2 * (step + 1)
        result = {"output": y.detach().cpu(), "dx": x.grad.cpu(), "dscores": scores.grad.cpu()}
        for i, expert in enumerate(experts):
            for name in ("gate_up", "down"):
                result[f"{i}.{name}"] = local_tensor(getattr(expert, name).grad).cpu()
        baseline = args.output / f"split_rank{rank}_step{step}.pt"
        if args.layout == "split":
            torch.save(result, baseline)
        else:
            expected = torch.load(baseline, weights_only=True)
            for name, tensor in result.items():
                if not tensor.isfinite().all() or not tensor.count_nonzero():
                    raise AssertionError(f"rank {rank} step {step}: invalid {name}")
                torch.testing.assert_close(tensor, expected[name], atol=0, rtol=0,
                                           msg=lambda msg: f"rank {rank} step {step} {name}: {msg}")
    stats = torch.tensor([runtime.statistics["max_live_weight_leases"], runtime.statistics["remote_experts_max"]],
                         device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    max_leases, remote_experts = stats.tolist()
    assert max_leases <= 2 and remote_experts > 0, (max_leases, remote_experts)
    if args.backend == "megakernel":
        runtime.close()
    dist.barrier()
    if rank == 0:
        print(f"PASS {args.backend} {args.layout}: 4 ranks, 2 training steps, max leases={max_leases}, "
              f"max remote experts={remote_experts}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("nonfused", "megakernel"), required=True)
    parser.add_argument("--layout", choices=("split", "whole"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path)
    run(parser.parse_args())
