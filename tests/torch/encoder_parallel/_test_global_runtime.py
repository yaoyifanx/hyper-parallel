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
"""CPU/Gloo worker for ViT ECP spanning two decoder DP samples."""

import hashlib
from dataclasses import fields
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.data.batching.get_batch import OmniParallelBatch
from hyper_parallel.distributed.encoder_parallel.batch_input import EncoderParallelInputAdapter
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.runtime import EncoderParallelRuntime
from hyper_parallel.models.deepseek_v41.adapter.distributed.encoder_parallel import (
    get_deepseek_v41_encoder_parallel_spec,
    replace_vision_attention,
)
from hyper_parallel.models.deepseek_v41.vision import (
    DeepseekV41VisionAligner,
    DeepseekV41VisionAttention,
    DeepseekV41VisionTower,
)
from hyper_parallel.models.replacement import (
    ModuleReplacementSpec,
    apply_module_replacements,
    compile_module_replacements,
)
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token


def _make_runtime(rank, encoder_cp_size, group_scope="global", vision_depth=1):
    """Build a tiny real ViT with eight divisible attention heads."""
    torch.manual_seed(10)
    config = SimpleNamespace(
        v41_vision_hidden_size=32, v41_vision_patch_size=2, v41_vision_num_attention_heads=8,
        v41_vision_rope_theta=10000.0, v41_vision_intermediate_size=64,
        v41_vision_num_hidden_layers=vision_depth, v41_vision_downsample_ratio=2, hidden_size=32,
    )
    model = nn.Module()
    model.model = nn.Module()
    model.model.vision = DeepseekV41VisionTower(config)
    model.model.aligner = DeepseekV41VisionAligner(config)
    for name in ("image_start", "image_end", "image_newline"):
        setattr(model.model, name, nn.Parameter(torch.zeros(32)))
    rule = ModuleReplacementSpec(
        match=("model.vision.blocks.*.attn",), factory=replace_vision_attention,
        module_type=DeepseekV41VisionAttention,
    )
    apply_module_replacements(model, compile_module_replacements(model, (rule,)), weights_mapping=[])
    mesh = init_device_mesh("cpu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"),
                            rank_list=tuple(range(8)), init_backend=True)
    context = SimpleNamespace(
        device_mesh=mesh, dp_size=2, dp_rank=rank // 4,
        cp_size=2, cp_rank=(rank % 4) // 2,
        tp_size=2, tp_rank=rank % 2, pp_size=1,
        sequence_parallel=True, loss_parallel=False,
    )
    runtime = EncoderParallelRuntime(
        config=EncoderParallelConfig(enabled=True, encoder_cp_size=encoder_cp_size, group_scope=group_scope),
        mesh_context=context, model=model, spec=get_deepseek_v41_encoder_parallel_spec(),
        execution_options={"micro_batch_size": 1, "compile_enabled": False,
                           "validate_placement": False, "record_image_identity": True},
    )
    return model, runtime


def _make_batch(owner, text_only_dp):
    """Give each decoder DP one image with a distinct patch count."""
    torch.manual_seed(20 + owner)
    height, width = (4, 4) if owner == 0 else (2, 4)
    llm_height, llm_width = height // 2, width // 2
    patch_count = height * width
    batch = {
        "input_ids": torch.zeros(1, 16, dtype=torch.long),
        "labels": torch.zeros(1, 16, dtype=torch.long),
        "token_types": torch.zeros(1, 16, dtype=torch.long),
    }
    if text_only_dp and owner == 1:
        return batch
    batch.update({
        "pixel_values": torch.randn(patch_count, 3, 2, 2),
        "image_patch_offsets": torch.tensor([0, patch_count]),
        "image_vit_grid_hw": torch.tensor([[height, width]]),
        "image_llm_grid_hw": torch.tensor([[llm_height, llm_width]]),
        "image_batch_indices": torch.tensor([0]),
        "image_token_starts": torch.tensor([2 if owner == 0 else 4]),
    })
    return batch


def _assert_plan(plan, batch, rank, encoder_cp_size, text_only_dp, group_scope):
    """Check global work keys, local shards, and source image identity."""
    if group_scope == "sample":
        expected_keys = () if text_only_dp and rank >= 4 else ((0, 0),)
        expected_buckets = 4 // encoder_cp_size
    else:
        expected_keys = ((0, 0),) if text_only_dp else ((0, 0), (1, 0))
        expected_buckets = 8 // encoder_cp_size
    assert plan.active and len(plan.buckets) == expected_buckets
    assert tuple(sorted(item.key for bucket in plan.buckets for item in bucket)) == expected_keys
    assert plan.local_patches is not None or not any(plan.buckets)
    assert "local_patches" not in {item.name for item in fields(plan)}
    canonical_with_image = rank % 4 == 0 and not (text_only_dp and rank == 4)
    if canonical_with_image:
        pixels = batch["pixel_values"]
        digest = hashlib.sha256()
        digest.update(f"{tuple(pixels.shape)}:{pixels.dtype}".encode())
        digest.update(pixels.contiguous().view(torch.uint8).numpy().tobytes())
        assert torch.equal(plan.source_image_digest, torch.tensor([list(digest.digest())], dtype=torch.uint8))
    else:
        assert plan.source_image_digest is None


def _assert_records(records, batch, model, rank, text_only_dp, group_scope):
    """Compare feature destinations and values with the local reference."""
    expected = (0, 0, 0, None, None, None, None, None)[rank] if text_only_dp else (
        0, 0, 0, None, None, 1, 1, None
    )[rank]
    if group_scope == "sample" and expected is not None:
        expected = 0
    assert tuple(item.owner_dp_rank for item, _ in records) == (() if expected is None else (expected,))
    if records:
        with torch.no_grad():
            height, width = (int(value) for value in batch["image_vit_grid_hw"][0])
            reference = model.model.aligner(
                model.model.vision(batch["pixel_values"], height, width), height, width,
            )
            assert torch.allclose(records[0][1], reference, atol=1e-5, rtol=1e-5)


def _assert_gradients(records, anchor, batch, model, rank, text_only_dp):
    """Compare distributed ViT and aligner gradients with full-image references."""
    loss = anchor + sum((feature.sum() for _, feature in records), torch.tensor(0.0))
    loss.backward()
    parameters = (*model.model.vision.parameters(), *model.model.aligner.parameters())
    candidate_gradients = []
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        candidate_gradients.append(parameter.grad.detach().clone())
    model.zero_grad(set_to_none=True)
    if rank == 0 or rank == 4 and not text_only_dp:
        height, width = (int(value) for value in batch["image_vit_grid_hw"][0])
        reference = model.model.aligner(
            model.model.vision(batch["pixel_values"], height, width), height, width,
        )
        reference.sum().mul(3 if rank == 0 else 2).backward()
    for parameter, candidate in zip(parameters, candidate_gradients):
        expected_gradient = torch.zeros_like(candidate) if parameter.grad is None else parameter.grad
        dist.all_reduce(candidate)
        dist.all_reduce(expected_gradient)
        assert torch.allclose(candidate, expected_gradient, atol=1e-4, rtol=1e-4), (
            f"rank={rank}; max_abs={(candidate - expected_gradient).abs().max().item()}"
        )


def _run_global_runtime(encoder_cp_size, text_only_dp, group_scope="global", vision_depth=1):
    """Route two decoder samples through the selected global ViT layout."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    model, runtime = _make_runtime(rank, encoder_cp_size, group_scope, vision_depth)
    batch = _make_batch(rank // 4, text_only_dp)
    omni_batch = OmniParallelBatch(
        mesh_context=runtime.mesh_context, device=torch.device("cpu"),
        runtime_input_adapter=EncoderParallelInputAdapter(runtime=runtime, inner=None),
    )
    omni_batch.source_only_pixel_values = True
    model_inputs, _ = omni_batch(iter((batch,)) if rank % 2 == 0 else None)
    assert "pixel_values" not in model_inputs
    plan = model_inputs["encoder_parallel_plan"]
    _assert_plan(plan, batch, rank, encoder_cp_size, text_only_dp, group_scope)
    records, anchor = plan.encode_and_route(model_inputs)
    _assert_records(records, batch, model, rank, text_only_dp, group_scope)
    _assert_gradients(records, anchor, batch, model, rank, text_only_dp)
    if rank == 0:
        print(f"{group_scope} ECP{encoder_cp_size} forward/backward passed", flush=True)
    dist.destroy_process_group()


def test_global_runtime_dp2_cp2_tp2_ecp8():
    """Route two DP images through one ECP8 ViT and backpropagate."""
    _run_global_runtime(8, False)


def test_global_runtime_dp2_cp2_tp2_ecp4():
    """Place one DP image in each of two ECP4 work buckets."""
    _run_global_runtime(4, False)


def test_global_runtime_ecp4_text_only_dp():
    """Keep a text-only decoder DP on an empty ECP4 bucket."""
    _run_global_runtime(4, True)


def test_omni_prefetch_error_all_ranks():
    """A source-side CPU batch error must reach its TP peer and other ranks."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    tp_group = dist.new_group(ranks=(0, 1))
    reader = OmniParallelBatch.__new__(OmniParallelBatch)
    reader.parallel_context = SimpleNamespace(
        build_on_rank=lambda: rank == 0, tp_world_size=2, tp_group=tp_group,
        cp_rank=0, distributed_enabled=True,
    )
    reader.cp_sharder = SimpleNamespace(shard=lambda batch: batch)
    reader.source_only_pixel_values = False
    reader.device = torch.device("cpu")
    source = {"input_ids": torch.ones(1, 4, dtype=torch.long)} if rank == 0 else None
    try:
        reader.prepare_step_sources(iter((source,)) if rank == 0 else None, 1, count_loss_token)
    except ValueError as exc:
        assert "Omni CPU batch prefetch failed" in str(exc)
        assert "Omni batch must contain labels" in str(exc)
    else:
        raise AssertionError("source error was not propagated to every rank")
    valid_sources = ({
        "input_ids": torch.zeros(1, count + 1, dtype=torch.long),
        "labels": torch.zeros(1, count + 1, dtype=torch.long),
    } for count in (2, 4, 7)) if rank == 0 else None
    batches, counts = reader.prepare_step_sources(valid_sources, 3, count_loss_token)
    assert len(batches) == 3
    assert int(counts["foundation_tokens"]) == 13
    assert (batches[0] is None) == (rank != 0)
    dist.destroy_process_group()


def test_sample_runtime_source_only_pixels_with_checkpoint():
    """Route source-only patches within each sample and recompute four ViT blocks."""
    _run_global_runtime(4, False, "sample", 4)


def test_sample_runtime_text_only_dp():
    """Keep an image-free sample on the same collective and gradient path."""
    _run_global_runtime(4, True, "sample")


def test_global_preflight_error_all_ranks():
    """Reject one decoder replica's corrupt image metadata on every rank."""
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    _, runtime = _make_runtime(rank, 8)
    batch = _make_batch(rank // 4, False)
    if rank == 2:
        batch["image_patch_offsets"] = torch.tensor([0, 15])
    omni_batch = OmniParallelBatch(
        mesh_context=runtime.mesh_context, device=torch.device("cpu"),
        runtime_input_adapter=EncoderParallelInputAdapter(runtime=runtime, inner=None),
    )
    omni_batch.source_only_pixel_values = True
    try:
        omni_batch(iter((batch,)) if rank % 2 == 0 else None)
    except ValueError as exc:
        assert "global encoder preflight failed on arena rank 2" in str(exc)
    else:
        raise AssertionError("corrupt image metadata was accepted")
    dist.destroy_process_group()
