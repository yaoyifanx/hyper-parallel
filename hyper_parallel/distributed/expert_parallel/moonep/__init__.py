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
"""MoonEP capacity planning, shared weights, and FSDP-driven replica prefetch.

Configure an already wrapped model with ``configure_moonep(model)``. Weight
slots allocate their maximum admitted capacity when first used; unused logical
slots retain no physical storage. Reader events protect concurrent prefetch and
reuse. Gradient accumulation is controlled by each executor's ``gradient_accumulation_fusion`` policy or its
model adapter's accumulation marker.
"""

from .checkpoint import wrap_moonep_checkpoint
from .runtime import MoonEPExecutor, MoonEPRuntime, configure_moonep

__all__ = ["MoonEPExecutor", "MoonEPRuntime", "configure_moonep", "wrap_moonep_checkpoint"]
