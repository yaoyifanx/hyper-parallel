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

namespace routed {

// Pointer and geometry values are checked by the Torch adapter before admission.
struct GroupLaunch {
  uint64_t input, output, scores, target_scores, saved, output_scores;
  uint64_t source_tokens, source_members, target_members, peer_offsets, row_slots;
  uint64_t input_ready, input_expected, output_ready, ring, signals, headers, progress;
  int64_t input_rows, output_rows, score_rows, output_score_rows, saved_rows, send_groups, receive_groups;
  int64_t epoch;
  uint64_t watchdog;
  int32_t rank, ranks, lanes, window_rows, header_words, hidden, top_k, quantum;
  int32_t phase, input_float, score_bf16, producers, experts, ready_tiles;
};

}  // namespace routed

extern "C" int RoutedExpertsGroupExchange(void *stream, const routed::GroupLaunch *args, int32_t workers);
