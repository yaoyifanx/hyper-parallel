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
"""Borrow the existing EP collective stream for every MoonEP payload role."""

from typing import Any

import torch
import torch.distributed as dist


def communication_stream(group: Any, device: torch.device) -> Any:
    """Wrap the group's existing stream without allocating an application stream.

    Args:
        group: Existing HCCL process group with the EP ranks in route order,
            or None to use the initialized WORLD group.
        device: Local NPU used by that process group.

    Returns:
        A Python stream wrapper shared by token, replica and replica-gradient
        submissions. Their compute dependencies remain explicit device events.
    """
    group = dist.group.WORLD if group is None else group
    backend = group._get_backend(device)
    stream_id = backend.get_coll_stream_id(device)
    current = torch.npu.current_stream(device)
    stream = torch.npu.Stream(stream_id=stream_id, device_index=device.index,
                              device_type=current.device_type)
    if stream.stream_id != stream_id:
        raise RuntimeError("communication wrapper does not reference the existing collective stream")
    return stream
