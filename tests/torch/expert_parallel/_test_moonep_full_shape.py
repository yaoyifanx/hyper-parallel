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
"""Actual GLM expert shapes and distant FP32 home-buffer output pointers."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu  # pylint: disable=unused-import

from hyper_parallel.core.multicore.modules.routed_experts import transport
from hyper_parallel.core.multicore.modules.routed_experts.compute import NativeExpertCompute
from hyper_parallel.core.multicore.modules.routed_experts.frame import ExpertState
from hyper_parallel.distributed.expert_parallel.moonep.numerics import grouped_weight_grad_add


def run(native_dir: Path) -> dict:
    """Check H4096/I1408, physical transpose and slots zero/63 against FP64 products."""
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(6001)
    device = torch.device("npu:0")
    libraries = tuple(native_dir / f"libhyper_parallel_routed_experts_{name}.so" for name in ("kernels", "torch"))
    transport.get_routed_experts_paths = lambda: libraries
    hidden, inner, homes = 4096, 1408, 64
    boundaries = (3, 20)
    selected = (0, 63)
    gate = (torch.randn(2, 2 * inner, hidden, device=device) * 0.025).bfloat16()
    down = (torch.randn(2, hidden, inner, device=device) * 0.025).bfloat16()
    home_gate = torch.zeros((homes, 2 * inner, hidden), device=device, dtype=torch.float32)
    home_down = torch.zeros((homes, hidden, inner), device=device, dtype=torch.float32)
    mapped = (tuple(home_gate[index] for index in selected), tuple(home_down[index] for index in selected))
    nf_gate = torch.zeros_like(gate, dtype=torch.float32)
    nf_down = torch.zeros_like(down, dtype=torch.float32)
    references = [torch.zeros_like(gate, device="cpu", dtype=torch.float64),
                  torch.zeros_like(down, device="cpu", dtype=torch.float64)]
    errors = []
    for _ in range(2):
        x = (torch.randn(20, hidden, device=device) * 0.1).bfloat16()
        z = (torch.randn(20, 2 * inner, device=device) * 0.1).bfloat16()
        dy = (torch.randn(20, hidden, device=device) * 0.1).bfloat16()
        compute = NativeExpertCompute(boundaries, device)
        work = compute.prepare_backward(dy, ExpertState(x, z), tuple(gate.transpose(1, 2)),
                                         tuple(down.transpose(1, 2)), pipeline=True,
                                         early_experts=(False, False), gradient_outputs=mapped,
                                         gradient_transposed=(True, True))
        compute.run_backward(work, lambda _role, _value: None, submit_return=lambda: None)
        groups = torch.tensor(boundaries, dtype=torch.int64, device=device)
        grouped_weight_grad_add(nf_gate, x, work.dz, groups, transposed=True)
        grouped_weight_grad_add(nf_down, work.activation, dy, groups, transposed=True)
        torch.npu.synchronize()
        pairs = ((work.dz.cpu().double(), x.cpu().double()),
                 (dy.cpu().double(), work.activation.cpu().double()))
        for reference, (left, right) in zip(references, pairs):
            start = 0
            for index, end in enumerate(boundaries):
                reference[index] += left[start:end].T @ right[start:end]
                start = end
        row = {"nonfused": [], "megakernel": []}
        for reference, actual_nf, actual_mega in zip(references, (nf_gate, nf_down), mapped):
            actual = torch.stack(tuple(value.cpu() for value in actual_mega)).double()
            mega_error = float((actual - reference).norm() / reference.norm().clamp_min(1e-30))
            nf_error = float((actual_nf.cpu().double() - reference).norm() / reference.norm().clamp_min(1e-30))
            if max(mega_error, nf_error) > 2e-5:
                raise AssertionError(f"full-shape FP32 accumulation errors: native={mega_error}, NF={nf_error}")
            row["nonfused"].append(nf_error)
            row["megakernel"].append(mega_error)
        errors.append(row)
    for tensor in (home_gate, home_down):
        for index in (1, 62):
            if int(torch.count_nonzero(tensor[index]).item()):
                raise AssertionError("mapped output wrote an unselected home expert")
    return {"hidden": hidden, "intermediate": inner, "home_buffer_experts": homes,
            "active_slots": selected, "microsteps": 2, "errors": errors,
            "home_gradient_bytes": home_gate.numel() * 4 + home_down.numel() * 4}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.native_dir), indent=2) + "\n")
