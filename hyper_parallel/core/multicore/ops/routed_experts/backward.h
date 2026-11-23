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

// Publish selected full-token dW early, then token dX, and finish the remaining
// dW in the same cube launch. External consumers join independent device flags.
extern "C" int RoutedExpertsBackwardTail(void *stream, void *dz, void *weights, void *dx_tasks,
    void *dx, void *dx_config, void *output_ready, void *x, void *activation, void *dv,
    void *gate_tasks, void *gate_gradient, void *gate_config,
    void *down_tasks, void *down_gradient, void *down_config, void *early_experts, void *gradient_ready,
    int32_t dx_count, int32_t gate_count, int32_t down_count,
    int32_t hidden, int32_t inner, int32_t experts, int32_t cores, bool transpose_b);

extern "C" int RoutedExpertsBackwardAccumulate(void *stream, void *dz, void *weights, void *dx_tasks,
    void *dx, void *dx_config, void *output_ready, void *x, void *activation, void *dv,
    void *gate_tasks, void *gate_gradient, void *gate_config,
    void *down_tasks, void *down_gradient, void *down_config, void *early_experts, void *gradient_ready,
    int32_t dx_count, int32_t gate_count, int32_t down_count,
    int32_t hidden, int32_t inner, int32_t experts, int32_t cores, bool transpose_b, bool gate_transposed, bool down_transposed);

extern "C" int RoutedExpertsWaitProducers(void *stream, void *flags, int32_t producers, int64_t watchdog);

extern "C" int RoutedExpertsBackwardTailW2First(void *stream, void *dz, void *weights, void *dx_tasks,
    void *dx, void *dx_config, void *output_ready, void *x, void *activation, void *dv,
    void *gate_tasks, void *gate_gradient, void *gate_config,
    void *down_tasks, void *down_gradient, void *down_config, void *early_experts, void *gradient_ready,
    int32_t dx_count, int32_t gate_count, int32_t down_count,
    int32_t hidden, int32_t inner, int32_t experts, int32_t cores, bool transpose_b);

extern "C" int RoutedExpertsBackwardAccumulateW2First(void *stream, void *dz, void *weights, void *dx_tasks,
    void *dx, void *dx_config, void *output_ready, void *x, void *activation, void *dv,
    void *gate_tasks, void *gate_gradient, void *gate_config,
    void *down_tasks, void *down_gradient, void *down_config, void *early_experts, void *gradient_ready,
    int32_t dx_count, int32_t gate_count, int32_t down_count,
    int32_t hidden, int32_t inner, int32_t experts, int32_t cores, bool transpose_b, bool gate_transposed, bool down_transposed);


extern "C" int RoutedExpertsMappedGradient(void *stream, void *input, void *derivative, void *tasks,
    void *outputs, void *config, void *priorities,
    int32_t count, int32_t features, int32_t columns, int32_t experts, int32_t cores);
