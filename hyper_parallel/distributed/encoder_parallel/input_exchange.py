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
"""Route source image patches to Encoder-CP workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed.encoder_parallel.contracts import EncoderSegment, EncoderWorkItem
from hyper_parallel.distributed.encoder_parallel.planning import build_segments


def _valid_segment(item: EncoderWorkItem, cp_rank: int, cp_size: int) -> EncoderSegment:
    """Reuse the packed ViT segment planner for one image destination."""
    return build_segments((item,), cp_rank, cp_size)[0]


def _local_source_descriptor(image_inputs, buckets, arena_rank, decoder_parent_size):
    """Validate this rank's canonical patch payload, if it owns images."""
    parent_rank = arena_rank % decoder_parent_size
    pixels = image_inputs.get("pixel_values")
    owned_items = tuple(item for bucket in buckets for item in bucket
                        if item.owner_dp_rank == arena_rank // decoder_parent_size)
    descriptor = None
    error = None
    if parent_rank == 0 and owned_items:
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            error = "canonical image source requires four-dimensional pixel_values"
        elif pixels.requires_grad:
            error = "patch transport does not support pixel_values gradients"
        elif pixels.shape[0] != max(item.payload_stop for item in owned_items):
            error = "canonical pixel_values patch count disagrees with image metadata"
        else:
            descriptor = (pixels.dtype, tuple(pixels.shape[1:]))
    return pixels, error, descriptor


def _source_descriptor(image_inputs, buckets, arena_rank, decoder_parent_size, arena_mesh):
    """Agree on the patch dtype and shape before tensor communication."""
    pixels, error, descriptor = _local_source_descriptor(
        image_inputs, buckets, arena_rank, decoder_parent_size,
    )
    descriptions = [None] * arena_mesh.size()
    if dist.is_initialized() and arena_mesh.size() > 1:
        dist.all_gather_object(descriptions, (error, descriptor), group=arena_mesh.get_group())
    else:
        descriptions[0] = (error, descriptor)
    failures = [(rank, value[0]) for rank, value in enumerate(descriptions) if value[0] is not None]
    if failures:
        raise ValueError(f"patch source rank {failures[0][0]}: {failures[0][1]}")
    source_descriptors = [value[1] for value in descriptions if value[1] is not None]
    if not source_descriptors:
        raise ValueError("active encoder plan has no image patch source")
    dtype, tail = source_descriptors[0]
    if any(value != (dtype, tail) for value in source_descriptors):
        raise ValueError("image patch dtype or shape differs across sources")
    return pixels, dtype, tail


def _pack_source_rows(pixels, buckets, encoder_cp_size, decoder_parent_size, arena_rank, tail, dtype, device):
    """Pack valid local-source patch rows in destination-rank order."""
    send_chunks = []
    input_splits = []
    for destination_rank in range(len(buckets) * encoder_cp_size):
        bucket = buckets[destination_rank // encoder_cp_size]
        cp_rank = destination_rank % encoder_cp_size
        chunks = []
        for item in bucket:
            if item.owner_dp_rank * decoder_parent_size != arena_rank:
                continue
            segment = _valid_segment(item, cp_rank, encoder_cp_size)
            if segment.local_valid_length:
                start = item.payload_start + segment.local_start
                chunks.append(pixels[start:start + segment.local_valid_length])
        input_splits.append(sum(chunk.shape[0] for chunk in chunks))
        send_chunks.extend(chunks)
    packed = torch.cat(send_chunks, dim=0) if send_chunks else torch.empty((0, *tail), dtype=dtype, device=device)
    return packed, input_splits


def _receive_splits(local_segments, decoder_parent_size, group_size):
    """Count rows expected from each canonical decoder source rank."""
    return [sum(segment.local_valid_length for segment in local_segments
                if not segment.is_dummy
                and segment.item.owner_dp_rank * decoder_parent_size == source_rank)
            for source_rank in range(group_size)]


def _order_received_rows(received, output_splits, local_segments, local_bucket,
                         arena_rank, encoder_cp_size, decoder_parent_size):
    """Restore packed ViT segment order after source-major all-to-all."""
    by_key = {}
    offset = 0
    for source_rank, split in enumerate(output_splits):
        source_offset = offset
        for item in local_bucket:
            if item.owner_dp_rank * decoder_parent_size != source_rank:
                continue
            segment = _valid_segment(item, arena_rank % encoder_cp_size, encoder_cp_size)
            stop = source_offset + segment.local_valid_length
            by_key[item.key] = received[source_offset:stop]
            source_offset = stop
        if source_offset != offset + split:
            raise ValueError("received patch rows disagree with image plan")
        offset += split
    ordered = [by_key[segment.item.key] for segment in local_segments if not segment.is_dummy]
    return torch.cat(ordered, dim=0) if ordered else received[:0]


def route_patch_shards(
        *, image_inputs: Mapping[str, Any], buckets: Sequence[Sequence[EncoderWorkItem]],
        local_segments: Sequence[EncoderSegment], encoder_cp_size: int,
        decoder_parent_size: int, arena_rank: int, arena_mesh: Any, device: torch.device,
) -> torch.Tensor:
    """Send each source image's valid patch rows to its assigned ECP ranks.

    Args:
        image_inputs: Local decoder sample's image tensors.
        buckets: Identical image work plan on every rank in the transport mesh.
        local_segments: This rank's planned ViT segments.
        encoder_cp_size: Number of ranks in one ViT sequence group.
        decoder_parent_size: CP times TP ranks per decoder sample.
        arena_rank: Local rank in the patch transport mesh.
        arena_mesh: Patch transport mesh, either one decoder sample or the global arena.
        device: Local model device used for empty sends and receives.

    Returns:
        Effective patches in local segment order, without padding rows.
    """
    group_size = arena_mesh.size()
    pixels, dtype, tail = _source_descriptor(
        image_inputs, buckets, arena_rank, decoder_parent_size, arena_mesh,
    )
    packed, input_splits = _pack_source_rows(
        pixels, buckets, encoder_cp_size, decoder_parent_size, arena_rank, tail, dtype, device,
    )
    output_splits = _receive_splits(local_segments, decoder_parent_size, group_size)
    local_bucket = buckets[arena_rank // encoder_cp_size]
    received = torch.empty((sum(output_splits), *tail), dtype=dtype, device=device)
    if dist.is_initialized() and group_size > 1:
        dist.all_to_all_single(received, packed, output_split_sizes=output_splits,
                               input_split_sizes=input_splits, group=arena_mesh.get_group())
    else:
        received.copy_(packed)
    return _order_received_rows(
        received, output_splits, local_segments, local_bucket,
        arena_rank, encoder_cp_size, decoder_parent_size,
    )
