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
"""One deterministic packed replica exchange per weight or gradient phase."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Fragment:
    """One complete expert matrix; wire role ``gate`` includes both gate/up projections."""

    source: int
    target: int
    expert: int
    role: str
    elements: int


@dataclass(frozen=True)
class LocalReplicaTransfer:
    """One rank's ordered matrix views and peer splits, in tensor elements."""

    sent: tuple[Fragment, ...]
    received: tuple[Fragment, ...]
    send_sizes: tuple[int, ...]
    receive_sizes: tuple[int, ...]


@dataclass(frozen=True)
class ReplicaTransfer:
    """The single global exchange, including participants with zero local splits."""

    fragments: tuple[Fragment, ...]
    ranks: int
    _local: dict[int, LocalReplicaTransfer] = field(default_factory=dict, compare=False, repr=False)

    def local(self, rank: int) -> LocalReplicaTransfer:
        """Return the cached local matrix views and their element-count peer splits.

        Args:
            rank: EP-local participant, including a rank with no sent/received matrices.

        Returns:
            A stable named record in the original global fragment order.
        """
        if rank in self._local:
            return self._local[rank]
        sent = tuple(fragment for fragment in self.fragments if fragment.source == rank)
        received = tuple(fragment for fragment in self.fragments if fragment.target == rank)
        send_sizes = tuple(sum(item.elements for item in sent if item.target == peer) for peer in range(self.ranks))
        receive_sizes = tuple(sum(item.elements for item in received if item.source == peer)
                              for peer in range(self.ranks))
        self._local[rank] = LocalReplicaTransfer(sent, received, send_sizes, receive_sizes)
        return self._local[rank]


def _check_transfer_allocation(allocation: torch.Tensor) -> tuple[int, int]:
    """Check the allocation table before inspecting complete matrix roles."""
    if (allocation.device.type != "cpu" or allocation.dtype != torch.int64 or allocation.ndim != 2
            or min(allocation.shape) < 1 or bool((allocation < 0).any())):
        raise ValueError("allocation must be a nonnegative CPU int64 [expert, rank] table")
    experts, ranks = allocation.shape
    if experts % ranks:
        raise ValueError("equal home expert ownership is required")
    return experts, ranks


def _check_transfer_roles(role_elements: tuple[tuple[str, int], ...]) -> None:
    """Validate role identity and element counts without reordering evaluation."""
    if (not role_elements or len({name for name, _ in role_elements}) != len(role_elements)
            or any(name not in ("gate", "down") or isinstance(size, bool) or not isinstance(size, int) or size < 1
                   for name, size in role_elements)):
        raise ValueError("each transfer role needs one positive integral matrix size")


def compile_replica_transfer(allocation: torch.Tensor, role_elements: tuple[tuple[str, int], ...], *,
                             reverse: bool = False) -> ReplicaTransfer | None:
    """Pack all complete matrices into a single peer-ordered collective.

    Args:
        allocation: Nonnegative CPU INT64 [expert, destination rank] placement.
        role_elements: Complete gate/up and down matrix element counts.
        reverse: Return replica gradients to their home owners when True.

    Returns:
        One descriptor, or None only when the entire group has no replicas.
        With any migration, zero-split ranks retain the same global descriptor.
        The sender reserves its actual fanout size; replica capacity does not
        impose an artificial per-transfer byte quota.
    """
    experts, ranks = _check_transfer_allocation(allocation)
    _check_transfer_roles(role_elements)
    transfers = []
    home_count = experts // ranks
    for expert, target in allocation.nonzero().tolist():
        owner = expert // home_count
        if target != owner:
            source, destination = (target, owner) if reverse else (owner, target)
            transfers.extend((source, destination, expert, role, elements) for role, elements in role_elements)
    fragments = tuple(Fragment(source, target, expert, role, elements)
                      for source, target, expert, role, elements in sorted(transfers))
    return ReplicaTransfer(fragments, ranks) if fragments else None


def copy_matrix(flat: torch.Tensor, matrix: torch.Tensor, *, unpack: bool = False) -> None:
    """Copy one complete matrix to or from a packed replica message."""
    if (matrix.ndim != 2 or flat.ndim != 1 or not flat.is_contiguous()
            or flat.numel() != matrix.numel()):
        raise ValueError("replica messages must contain a complete matrix")
    segment = flat.view(matrix.shape)
    if unpack:
        matrix.copy_(segment)
    else:
        segment.copy_(matrix)
