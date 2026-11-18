// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <cstdint>

namespace hyper_parallel::multicore::routed_experts {
constexpr int64_t kOrderedSumFields = 6;
constexpr int64_t kOrderedSumTile = 8192;
}

extern "C" int RoutedExpertsOrderedSum(void *stream, void *metadata, int64_t groups,
                                      int64_t tiles, int32_t cores);
