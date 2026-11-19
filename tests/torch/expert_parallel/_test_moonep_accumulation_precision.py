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
"""Single-NPU acceptance against a CPU FP64 expert-gradient oracle.

Run this worker directly in an activated CANN environment. The native payload
path is supplied explicitly so the test cannot validate a stale kernel build.
"""
import argparse
import json
from pathlib import Path

import torch
import torch_npu

from hyper_parallel.core.multicore.modules.routed_experts import transport
from hyper_parallel.core.multicore.modules.routed_experts.compute import NativeExpertCompute
from hyper_parallel.core.multicore.modules.routed_experts.frame import ExpertState
from hyper_parallel.core.fully_shard.hsdp_param import pack_reduce_scatter_input
from hyper_parallel.distributed.expert_parallel.moonep.numerics import grouped_weight_grad_add


def relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Return normalized L2 error using FP64 CPU arithmetic."""
    return float((actual.double() - reference).norm() / reference.norm().clamp_min(1e-30))


def run(native_dir: Path) -> dict:
    """Validate both layouts, nonzero prior g, replicas, empty groups and dX."""
    torch.npu.set_device(0)
    torch.manual_seed(20261119)
    device = torch.device("npu:0")
    libraries = tuple(native_dir / f"libhyper_parallel_routed_experts_{name}.so" for name in ("kernels", "torch"))
    if not all(value.is_file() for value in libraries):
        raise FileNotFoundError("both freshly built routed-experts libraries are required")
    transport.get_routed_experts_paths = lambda: libraries
    results = {"packing": [], "nonfused": [], "megakernel": [], "remote_add": []}
    for source in (torch.randn(4, 12, 8, device=device).bfloat16(),):
        for source in (source, source.transpose(1, 2)):
            for dim in (0, 1, 2):
                packed = pack_reduce_scatter_input(source, torch.float32, 2, dim)
                expected = torch.cat(source.float().chunk(2, dim=dim), dim=0).flatten()
                torch.testing.assert_close(packed, expected, rtol=0, atol=0)
                results["packing"].append({"dim": dim, "stride": source.stride(), "exact": True})
    for transpose in (False, True):
        shape = (3, 64, 32) if not transpose else (3, 32, 64)
        initial = torch.randn(shape) * 0.01
        target = initial.to(device)
        reference = initial.double().clone()
        bf16_reference = initial.bfloat16().float()
        rounded_dw_fp32_accum = initial.clone()
        for counts in ([3, 0, 257], [0, 17, 1], [0, 0, 0]):
            xx = (torch.randn(sum(counts), 64) * 0.1).bfloat16()
            yy = (torch.randn(sum(counts), 32) * 0.1).bfloat16()
            grouped_weight_grad_add(target, xx.to(device), yy.to(device),
                                    torch.tensor(counts, device=device).cumsum(0), transposed=transpose)
            start = 0
            for index, count in enumerate(counts):
                value = xx[start:start + count].double().T @ yy[start:start + count].double()
                value = value.T if transpose else value
                reference[index] += value
                rounded_dw_fp32_accum[index].add_(value.bfloat16().float())
                bf16_reference[index] = (bf16_reference[index] + value.bfloat16().float()).bfloat16().float()
                start += count
        actual = target.cpu()
        error, old_error = relative_error(actual, reference), relative_error(bf16_reference, reference)
        if error > 2e-5 or error > old_error:
            raise AssertionError(f"grouped matmul-add FP32 error {error} exceeds reference {old_error}")
        results["nonfused"].append({"transposed": transpose, "fp32_relative_l2": error,
                                     "bf16_accumulation_relative_l2": old_error,
                                     "bf16_dw_fp32_accum_relative_l2": relative_error(
                                         rounded_dw_fp32_accum, reference)})
    for dtype in (torch.bfloat16, torch.float32):
        for transposed in (False, True):
            target = torch.full((32, 64), 1024.0, dtype=torch.float32, device=device)
            shape = (64, 32) if transposed else (32, 64)
            remote = torch.full(shape, 0.00390625, dtype=dtype, device=device)
            view = remote.T if transposed else remote
            target.add_(view)
            target.add_(view)
            torch.testing.assert_close(target.cpu(), torch.full((32, 64), 1024.0078125), rtol=0, atol=0)
            results["remote_add"].append({"dtype": str(dtype), "transposed": transposed, "exact": True})
    boundaries = (3, 260, 277)
    for transpose in (False, True):
        hidden, inner = 64, 32
        gate = torch.randn(3, hidden, inner * 2).bfloat16().to(device) * 0.05
        down = torch.randn(3, inner, hidden).bfloat16().to(device) * 0.05
        # Real GLM weights expose logical transposed views of physical storage.
        if transpose:
            gate = gate.transpose(1, 2).contiguous().transpose(1, 2)
            down = down.transpose(1, 2).contiguous().transpose(1, 2)
        gshape = (3, inner * 2, hidden) if transpose else (3, hidden, inner * 2)
        dshape = (3, hidden, inner) if transpose else (3, inner, hidden)
        g = torch.randn(gshape, device=device) * 0.01
        d = torch.randn(dshape, device=device) * 0.01
        references = [g.cpu().double(), d.cpu().double()]
        # Expert 1 represents a replica: its independent output overwrites each microstep.
        replica = 1
        errors = []
        for _ in range(3):
            x = (torch.randn(boundaries[-1], hidden, device=device) * 0.1).bfloat16()
            z = (torch.randn(boundaries[-1], inner * 2, device=device) * 0.1).bfloat16()
            dy = (torch.randn_like(x.float()) * 0.1).bfloat16()
            compute = NativeExpertCompute(boundaries, device)
            work = compute.prepare_backward(dy, ExpertState(x, z), tuple(gate), tuple(down), pipeline=True,
                                             early_experts=(False, True, False),
                                             gradient_outputs=(tuple(g), tuple(d)),
                                             gradient_transposed=(transpose, transpose))
            compute.run_backward(work, lambda _role, _value: None, submit_return=lambda: None)
            torch.npu.synchronize()
            pairs = ((x.cpu().double(), work.dz.cpu().double()),
                     (work.activation.cpu().double(), dy.cpu().double()))
            for role, (left, right) in enumerate(pairs):
                start = 0
                for expert, end in enumerate(boundaries):
                    value = left[start:end].T @ right[start:end]
                    value = value.T if transpose else value
                    if expert == replica:
                        references[role][expert] = value
                    else:
                        references[role][expert] += value
                    start = end
            for actual, reference in zip((g, d), references):
                error = relative_error(actual.cpu(), reference)
                if error > 2e-5:
                    raise AssertionError(f"mapped native FP32 wgrad error {error}")
                errors.append(error)
            dx_reference = []
            start = 0
            for expert, end in enumerate(boundaries):
                dx_reference.append(work.dz[start:end].cpu().double() @ gate[expert].cpu().double().T)
                start = end
            dx_error = relative_error(work.output.cpu(), torch.cat(dx_reference))
            if dx_error > 0.005:
                raise AssertionError(f"native dX relative error {dx_error}")
        results["megakernel"].append({"transposed": transpose, "microsteps": 3,
                                       "max_fp32_relative_l2": max(errors), "dx_relative_l2": dx_error})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.native_dir), indent=2) + "\n")
