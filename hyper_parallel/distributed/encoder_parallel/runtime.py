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
"""Per-batch visual encoder planning and packed execution."""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import

from hyper_parallel.core.dtensor._utils import differentiable_variable_all_gather
from hyper_parallel.distributed.encoder_parallel.attention import (
    EncoderAttentionContext,
    activate_encoder_attention,
)
from hyper_parallel.distributed.encoder_parallel.config import EncoderParallelConfig
from hyper_parallel.distributed.encoder_parallel.contracts import (
    EncoderFeature,
    EncoderParallelAdapterSpec,
    EncoderSegment,
    EncoderWorkItem,
)
from hyper_parallel.distributed.encoder_parallel.exchange import (
    FeatureExchangeMode,
    route_features,
    select_feature_exchange,
    unpack_records,
)
from hyper_parallel.distributed.encoder_parallel.input_exchange import route_patch_shards
from hyper_parallel.distributed.encoder_parallel.mesh import derive_reshaped_submesh
from hyper_parallel.distributed.encoder_parallel.planning import (
    DecoderSequenceLayout,
    assign_lpt,
    build_destinations,
    build_global_destinations,
    build_segments,
)


def _invalid_head_count(count: Any) -> bool:
    """Reject booleans, nonintegers, and zero-sized QKV head axes."""
    return isinstance(count, bool) or not isinstance(count, int) or count < 1


def validate_ulysses_head_divisibility(*, encoder_cp_size: int, head_counts: Mapping[str, int],
                                       attention_pattern: str, model: nn.Module) -> None:
    """Check every Q/K/V head count before deriving any Encoder-CP group."""
    required = ("query", "key", "value")
    missing = [name for name in required if name not in head_counts]
    invalid = {name: head_counts[name] for name in required if name in head_counts
               and _invalid_head_count(head_counts[name])}
    non_divisible = {name: head_counts[name] for name in required if name in head_counts
                     and name not in invalid and head_counts[name] % encoder_cp_size}
    if missing or invalid or non_divisible:
        matched = sorted(name for name, _ in model.named_modules()
                         if fnmatch.fnmatchcase(name, attention_pattern))
        raise ValueError(
            "Encoder-CP/Ulysses head divisibility check failed; "
            "constraint=head_count % encoder_cp_size == 0 for every Q/K/V; "
            "config_path=accelerator.encoder_parallel.encoder_cp_size; "
            f"encoder_cp_size={encoder_cp_size}; head_counts={dict(head_counts)}; "
            f"missing={missing}; invalid_values={invalid}; non_divisible={non_divisible}; "
            f"attention_pattern={attention_pattern!r}; matched_modules={matched}"
        )


def validate_attention_replacement(model: nn.Module, spec: EncoderParallelAdapterSpec) -> frozenset[str]:
    """Require an exact structural replacement for every visual attention.

    Args:
        model: Final model after module replacements.
        spec: Model-owned attention matching and replacement contract.
    """
    expected = frozenset(name for name, _ in model.named_modules()
                         if fnmatch.fnmatchcase(name, spec.attention_pattern))
    recorded = set(getattr(model, "_hp_replaced_module_fqns", ()) or ())
    recorded_visual = {name for name in recorded if fnmatch.fnmatchcase(name, spec.attention_pattern)}
    missing = sorted(expected - recorded_visual)
    unexpected = sorted(recorded_visual - expected)
    wrong_type = sorted(name for name in expected
                        if not isinstance(model.get_submodule(name), spec.replacement_attention_type))
    if not expected or missing or unexpected or wrong_type:
        raise RuntimeError(
            "incomplete vision attention replacement: "
            f"expected={len(expected)}; missing={missing}; unexpected={unexpected}; wrong_type={wrong_type}"
        )
    return expected


def validate_encoder_execution_options(options: Mapping[str, Any], bridge: Any) -> None:
    """Reject unsupported execution modes before creating Encoder groups.

    Args:
        options: Trainer microbatch, compile, and placement settings.
        bridge: Model-bound validator for vision-specific restrictions.
    """
    bridge.validate_training(options)
    if options.get("compile_enabled"):
        raise NotImplementedError("v1 requires compile.enabled=false")
    if options.get("validate_placement"):
        raise NotImplementedError("v1 requires model.validate_placement=false")


def _canonical_global_items(descriptions: list[Any], dp_size: int,
                            parent_size: int) -> tuple[list[EncoderWorkItem], dict[int, int]]:
    """Validate decoder replicas and return one image task set per DP."""
    global_items = []
    cp_lengths = {}
    for owner in range(dp_size):
        replicas = [entry for entry in descriptions if entry[0] == owner]
        if len(replicas) != parent_size or {entry[1] for entry in replicas} != set(range(parent_size)):
            raise ValueError(f"decoder DP {owner} has incomplete image metadata replicas")
        canonical = next(entry for entry in replicas if entry[1] == 0)
        if any(entry[2:] != canonical[2:] for entry in replicas):
            raise ValueError(f"image metadata differs across decoder parent ranks: dp={owner}")
        cp_lengths[owner] = canonical[2]
        global_items.extend(replace(item, owner_dp_rank=owner) for item in canonical[3])
    return global_items, cp_lengths


@dataclass
class EncoderParallelPlan:
    """One microbatch of derived visual work and feature routing."""

    runtime: "EncoderParallelRuntime"
    active: bool
    decoder_layout: DecoderSequenceLayout
    buckets: tuple[tuple[EncoderWorkItem, ...], ...]
    destinations: dict[Any, tuple[int, ...]]
    local_segments: tuple[EncoderSegment, ...]
    k_fsdp_max: int
    source_image_digest: torch.Tensor | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        """Keep layout-dependent patch shards out of logical input identity."""
        self.local_patches: torch.Tensor | None = None

    def prepare_patches(self, image_inputs: Mapping[str, Any]) -> None:
        """Route source pixels before model forward so the full payload can be released.

        Args:
            image_inputs: Decoder-local image metadata and optional source pixels.
        """
        if not self.active or not any(self.buckets):
            return
        runtime = self.runtime
        global_scope = runtime.config.group_scope == "global"
        pixels = image_inputs.get("pixel_values")
        if runtime.record_image_identity and runtime.parent_rank == 0 and torch.is_tensor(pixels):
            digest = hashlib.sha256()
            digest.update(f"{tuple(pixels.shape)}:{pixels.dtype}".encode())
            digest.update(pixels.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            self.source_image_digest = torch.tensor([list(digest.digest())], dtype=torch.uint8)
        self.local_patches = route_patch_shards(
            image_inputs=image_inputs, buckets=self.buckets, local_segments=self.local_segments,
            encoder_cp_size=runtime.config.encoder_cp_size,
            decoder_parent_size=runtime.parent_size,
            arena_rank=runtime.arena_rank if global_scope else runtime.parent_rank,
            arena_mesh=runtime.global_mesh if global_scope else runtime.parent_mesh,
            device=next(runtime.model.parameters()).device,
        )

    def _gather_features(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """All-gather each local segment and trim dummy padding."""
        gathered = []
        offset = 0
        for segment in self.local_segments:
            local = hidden[offset:offset + segment.local_padded_length]
            offset += segment.local_padded_length
            full = differentiable_variable_all_gather(
                local, output_splits=[segment.local_padded_length] * self.runtime.config.encoder_cp_size,
                group=self.runtime.encoder_cp_mesh.get_group(),
            ) if self.runtime.config.encoder_cp_size > 1 else local
            gathered.append(full if segment.is_dummy else full[:segment.item.sequence_length])
        return tuple(gathered)

    def _execute_local(self, image_inputs: Mapping[str, Any]
                       ) -> tuple[tuple[EncoderFeature, ...], torch.Tensor, torch.Tensor]:
        """Run the local ViT pass and build routed feature records."""
        bridge = self.runtime.bridge
        context = EncoderAttentionContext(
            self.local_segments,
            self.runtime.encoder_cp_mesh,
            bridge.head_counts,
            self.runtime.expected,
        )
        with activate_encoder_attention(context):
            hidden = bridge.encode_local(image_inputs, self.local_segments)
        context.assert_exactly_once()
        full_features = self._gather_features(hidden)
        records = []
        anchors = [feature.sum() * 0.0 for feature in full_features]
        prototype = None
        for slot in range(self.k_fsdp_max):
            segment = self.local_segments[slot] if slot < len(self.local_segments) else None
            real = segment is not None and not segment.is_dummy and self.runtime.encoder_cp_rank == 0
            if real:
                feature = bridge.project(full_features[slot], segment.item)
                if feature.ndim != 2 or feature.shape[0] != segment.item.output_rows:
                    raise ValueError(
                        "aligner output does not match planned rows; "
                        f"item_id={segment.item.item_id}; actual_shape={tuple(feature.shape)}; "
                        f"expected_rows={segment.item.output_rows}"
                    )
                records.append((segment.item, feature))
            else:
                feature = bridge.project(hidden, None)
                anchors.append(feature.sum() * 0.0)
            prototype = feature
        if prototype is None:
            raise RuntimeError("active plan must have an aligner slot")
        anchor = torch.stack(anchors).sum() if anchors else prototype.sum() * 0.0
        return tuple(records), prototype, anchor

    def encode_and_route(self, image_inputs: Mapping[str, Any]
                         ) -> tuple[list[EncoderFeature], torch.Tensor | None]:
        """Run packed ViT and route aligned features to this decoder owner.

        Args:
            image_inputs: Original image tensors shared across the parent ranks.

        Returns:
            Routed image records and a zero-valued gradient anchor, or no work.
        """
        if not self.active:
            return [], None
        if self.local_patches is not None:
            image_inputs = {"local_patches": self.local_patches}
        elif self.runtime.config.group_scope == "global":
            local_patches = route_patch_shards(
                image_inputs=image_inputs, buckets=self.buckets, local_segments=self.local_segments,
                encoder_cp_size=self.runtime.config.encoder_cp_size,
                decoder_parent_size=self.runtime.parent_size, arena_rank=self.runtime.arena_rank,
                arena_mesh=self.runtime.global_mesh, device=next(self.runtime.model.parameters()).device,
            )
            image_inputs = {"local_patches": local_patches}
        records, prototype, local_anchor = self._execute_local(image_inputs)
        mode = self.runtime.feature_exchange_mode
        exchange_mesh = self.runtime.tp_mesh if mode in (
            FeatureExchangeMode.TP_VARIABLE_ALL_GATHER, FeatureExchangeMode.TP_VARIABLE_A2A
        ) else (self.runtime.global_mesh if mode is FeatureExchangeMode.GLOBAL_VARIABLE_A2A
                else self.runtime.parent_mesh)
        rows, received_items = route_features(
            mode=mode, producer_records=records, prototype=prototype,
            parent_rank=self.runtime.arena_rank if mode is FeatureExchangeMode.GLOBAL_VARIABLE_A2A
            else self.runtime.parent_rank, destinations=self.destinations,
            buckets=self.buckets, encoder_cp_size=self.runtime.config.encoder_cp_size, exchange_mesh=exchange_mesh,
        )
        unpacked = unpack_records(rows, received_items)
        return list(unpacked), local_anchor + rows.sum() * 0.0


class EncoderParallelRuntime:
    """Own static topology and create one plan per Omni microbatch."""

    def __init__(self, *, config: EncoderParallelConfig, mesh_context: Any,
                 model: nn.Module, spec: EncoderParallelAdapterSpec,
                 execution_options: Mapping[str, Any]) -> None:
        """Validate the static contract and derive Encoder-CP groups."""
        self.config = config
        self.mesh_context = mesh_context
        self.model = model
        self.spec = spec
        self.bridge = spec.bind(model)
        self.record_image_identity = bool(execution_options.get("record_image_identity", False))
        validate_encoder_execution_options(
            {**execution_options, "loss_parallel": mesh_context.loss_parallel}, self.bridge,
        )
        parent_size = mesh_context.cp_size * mesh_context.tp_size
        if mesh_context.pp_size != 1:
            raise NotImplementedError("v1 requires pp_size=1")
        global_scope = config.group_scope == "global"
        arena_size = parent_size * int(getattr(mesh_context, "dp_size", 1))
        domain_size = arena_size if global_scope else parent_size
        if domain_size % config.encoder_cp_size:
            if global_scope:
                raise ValueError("encoder_cp_size must divide dp_size*cp_size*tp_size")
            raise ValueError("encoder_cp_size must divide cp_size*tp_size")
        validate_ulysses_head_divisibility(encoder_cp_size=config.encoder_cp_size,
                                           head_counts=self.bridge.head_counts,
                                           attention_pattern=spec.attention_pattern, model=model)
        self.expected = validate_attention_replacement(model, spec)
        self.parent_rank = mesh_context.cp_rank * mesh_context.tp_size + mesh_context.tp_rank
        self.parent_size = parent_size
        self.arena_rank = int(getattr(mesh_context, "dp_rank", 0)) * parent_size + self.parent_rank
        domain_rank = self.arena_rank if global_scope else self.parent_rank
        self.encoder_cp_rank = domain_rank % config.encoder_cp_size
        self.encoder_dp_rank = domain_rank // config.encoder_cp_size
        self.encoder_dp_size = domain_size // config.encoder_cp_size
        self.parent_mesh = mesh_context.device_mesh[("cp", "tp")].flatten("encoder_parent")
        encoder_mesh = derive_reshaped_submesh(
            mesh_context.device_mesh, source_dims=("dp", "cp", "tp") if global_scope else ("cp", "tp"),
            target_sizes=(self.encoder_dp_size, config.encoder_cp_size),
            target_names=("encoder_dp", "encoder_cp"),
            parent_name="encoder_global_parent" if global_scope else "encoder_parent",
        )
        self.encoder_cp_mesh = encoder_mesh["encoder_cp"]
        self.tp_mesh = mesh_context.device_mesh["tp"]
        self.global_mesh = mesh_context.device_mesh.flatten("encoder_global")
        self.feature_exchange_mode = (FeatureExchangeMode.GLOBAL_VARIABLE_A2A if global_scope
                                      else select_feature_exchange(mesh_context))

    def _finalize_global_plan(self, items: tuple[EncoderWorkItem, ...],
                              layout: DecoderSequenceLayout) -> EncoderParallelPlan:
        """Gather one decoder sample description per DP and reuse local planning."""
        local = (int(self.mesh_context.dp_rank), self.parent_rank, layout.cp_local_length, items)
        descriptions = [None] * self.global_mesh.size()
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_gather_object(descriptions, local, group=self.global_mesh.get_group())
        else:
            descriptions[0] = local
        global_items, cp_lengths = _canonical_global_items(
            descriptions, int(self.mesh_context.dp_size), self.parent_size,
        )
        buckets = assign_lpt(global_items, self.encoder_dp_size, self.config.encoder_cp_size)
        layouts = {
            rank: DecoderSequenceLayout.build(
                mesh_context=self.mesh_context, cp_local_length=cp_lengths[rank // self.parent_size],
                cp_rank=(rank % self.parent_size) // self.mesh_context.tp_size,
                tp_rank=rank % self.mesh_context.tp_size,
            ) for rank in range(self.global_mesh.size())
        }
        destinations = build_global_destinations(global_items, layouts, self.parent_size)
        return EncoderParallelPlan(
            runtime=self, active=bool(global_items), decoder_layout=layout,
            buckets=buckets, destinations=destinations,
            local_segments=build_segments(buckets[self.encoder_dp_rank], self.encoder_cp_rank,
                                          self.config.encoder_cp_size),
            k_fsdp_max=max(1, *(len(bucket) for bucket in buckets)),
        )

    def preflight_batch(self, batch: Mapping[str, Any]
                        ) -> tuple[tuple[EncoderWorkItem, ...], DecoderSequenceLayout]:
        """Validate local metadata and SP geometry before control collectives.

        Args:
            batch: Device batch with decoder tokens and optional image fields.
        """
        layout = DecoderSequenceLayout.build(
            mesh_context=self.mesh_context, cp_local_length=int(batch["input_ids"].shape[1])
        )
        present = [batch.get(name) is not None for name in self.bridge.input_fields]
        if not present[0] and any(present[1:]):
            if self.parent_rank == 0:
                raise ValueError("canonical image source requires pixel_values")
            missing = [name for name in self.bridge.input_fields[1:] if batch.get(name) is None]
            if missing:
                raise ValueError(f"incomplete image metadata bundle: missing={missing}")
            return tuple(self.bridge.parse_images_metadata(batch)), layout
        if any(present) and not all(present):
            missing = [name for name in self.bridge.input_fields if batch.get(name) is None]
            raise ValueError(f"incomplete image metadata bundle: missing={missing}")
        return tuple(self.bridge.parse_images(batch)), layout

    def agree_or_raise(self, phase: str, error: Exception | None) -> None:
        """Make all arena ranks leave a failed local input phase together.

        Args:
            phase: Name of the completed local input phase.
            error: Local exception, if this rank failed the phase.
        """
        if self.config.group_scope != "global":
            if error is not None:
                raise error
            return
        group = self.global_mesh.get_group()
        failed = torch.tensor(int(error is not None), device=next(self.model.parameters()).device)
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
        if not bool(failed.item()):
            return
        local_message = None if error is None else (self.arena_rank, f"{type(error).__name__}: {error}")
        messages = [None] * self.global_mesh.size()
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_gather_object(messages, local_message, group=group)
        else:
            messages[0] = local_message
        failing_rank, diagnosis = min(message for message in messages if message is not None)
        raise ValueError(f"global encoder {phase} failed on arena rank {failing_rank}: {diagnosis}")

    def finalize_plan(self, items: tuple[EncoderWorkItem, ...],
                      layout: DecoderSequenceLayout) -> EncoderParallelPlan:
        """Agree on active mode, then assign images and exact decoder owners.

        Args:
            items: Locally validated image jobs.
            layout: Locally validated decoder token ownership.
        """
        if self.config.group_scope == "global":
            return self._finalize_global_plan(items, layout)
        marker = torch.tensor(int(bool(items)), device=self.model.device if hasattr(self.model, "device")
                              else next(self.model.parameters()).device)
        if dist.is_initialized() and self.parent_mesh.size() > 1:
            signature = tuple(
                (item.item_id, item.batch_index, item.sequence_length, item.output_rows,
                 item.decoder_token_start, item.decoder_token_length, item.payload_start, item.payload_stop,
                 tuple(sorted(item.metadata.items())))
                for item in items
            )
            digest = hashlib.sha256(repr((layout.cp_local_length, signature)).encode()).digest()
            digest_value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
            local_digest = torch.tensor(digest_value, dtype=torch.int64, device=marker.device)
            peer_digests = [torch.empty_like(local_digest) for _ in range(self.parent_mesh.size())]
            dist.all_gather(peer_digests, local_digest, group=self.parent_mesh.get_group())
            if any(int(peer.item()) != digest_value for peer in peer_digests):
                raise ValueError("image metadata differs across decoder parent ranks")
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_reduce(marker, op=dist.ReduceOp.MAX, group=self.global_mesh.get_group())
        active = bool(marker.item())
        buckets = assign_lpt(items, self.encoder_dp_size, self.config.encoder_cp_size)
        layouts = {rank: DecoderSequenceLayout.build(mesh_context=self.mesh_context,
                                                     cp_local_length=layout.cp_local_length,
                                                     cp_rank=rank // self.mesh_context.tp_size,
                                                     tp_rank=rank % self.mesh_context.tp_size)
                   for rank in range(self.mesh_context.cp_size * self.mesh_context.tp_size)}
        destinations = build_destinations(items, layouts)
        segments = build_segments(buckets[self.encoder_dp_rank], self.encoder_cp_rank,
                                  self.config.encoder_cp_size)
        slot_count = torch.tensor(len(buckets[self.encoder_dp_rank]), device=marker.device)
        if dist.is_initialized() and self.global_mesh.size() > 1:
            dist.all_reduce(slot_count, op=dist.ReduceOp.MAX, group=self.global_mesh.get_group())
        return EncoderParallelPlan(
            runtime=self,
            active=active, decoder_layout=layout, buckets=buckets,
            destinations=destinations, local_segments=segments,
            k_fsdp_max=max(1, int(slot_count.item())),
        )


def build_encoder_parallel_runtime(*, config: EncoderParallelConfig, mesh_context: Any,
                                   model: nn.Module | None, model_adapter_spec: Any,
                                   execution_options: Mapping[str, Any] | None) -> EncoderParallelRuntime | None:
    """Create the model-neutral runtime only when the feature is enabled."""
    if not config.enabled:
        return None
    if model is None or model_adapter_spec is None or model_adapter_spec.encoder_parallel is None:
        raise ValueError("enabled encoder parallel requires a model adapter provider")
    spec = model_adapter_spec.encoder_parallel()
    if not isinstance(spec, EncoderParallelAdapterSpec):
        raise TypeError("encoder_parallel provider returned an invalid spec")
    return EncoderParallelRuntime(config=config, mesh_context=mesh_context, model=model,
                                  spec=spec, execution_options=execution_options or {})
