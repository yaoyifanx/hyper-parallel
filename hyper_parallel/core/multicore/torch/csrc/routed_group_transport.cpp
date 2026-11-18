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
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <algorithm>
#include <cstdint>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "group_transport.h"

namespace {
void Tensor(const at::Tensor &value, at::ScalarType dtype, c10::Device device) {
  TORCH_CHECK(value.scalar_type() == dtype && value.device() == device && value.is_contiguous(),
              "group exchange operands must be contiguous with the declared dtype on one device");
}

void Validate(at::TensorList inputs, at::TensorList outputs, at::TensorList tables,
              const at::Tensor &ring, const at::Tensor &signals, const at::Tensor &headers,
              const at::Tensor &progress, at::IntArrayRef config,
              const c10::optional<at::Tensor> &ready, const c10::optional<at::Tensor> &expected,
              const c10::optional<at::Tensor> &published) {
  TORCH_CHECK(inputs.size() == 4 && outputs.size() == 2 && tables.size() == 5 && config.size() == 12,
              "group exchange requires value/score/prob/V, vector/scalar outputs, five maps and twelve controls");
  const auto device = inputs[0].device();
  TORCH_CHECK(config[0] >= 1 && config[0] <= 4 && config[1] >= 0 && config[1] < config[2] &&
                config[2] >= 1 && config[2] < (1LL << 30) && config[3] >= 1 && config[3] <= 2 &&
                config[4] >= 1 && config[4] <= 1024 && config[5] >= 1 && config[5] <= config[4] &&
                config[6] >= 1 && (config[7] == 0 || config[7] == 1) && config[8] >= 1 &&
                config[8] <= config[2] * config[3] && config[9] > 0 && config[10] >= 0 && config[11] >= 0,
              "group exchange phase, topology, window, epoch and worker controls are invalid");
  const int64_t phase = config[0], ranks = config[2], lanes = config[3], rows = config[4], quantum = config[5];
  const bool expand = phase == 1 || phase == 3;
  const auto value_type = inputs[0].scalar_type();
  TORCH_CHECK(value_type == at::kBFloat16 || (phase == 3 && value_type == at::kFloat),
              "group exchange requires BF16 experts; FP32 upstream G remains FP32");
  Tensor(inputs[0], value_type, device);
  TORCH_CHECK(inputs[0].dim() == 2 && inputs[0].size(1) > 0 && inputs[0].size(1) <= 32768 &&
                inputs[0].size(1) % 16 == 0, "group exchange hidden width must be a positive multiple of sixteen");
  const int64_t hidden = inputs[0].size(1);
  const auto score_type = phase == 1 && config[7] ? at::kBFloat16 : at::kFloat;
  Tensor(inputs[1], score_type, device);
  Tensor(inputs[2], at::kFloat, device);
  Tensor(inputs[3], at::kBFloat16, device);
  Tensor(outputs[0], expand ? at::kBFloat16 : at::kFloat, device);
  Tensor(outputs[1], at::kFloat, device);
  TORCH_CHECK(outputs[0].dim() == 2 && outputs[0].size(1) == hidden && inputs[1].dim() == 1 &&
                inputs[2].dim() == 1 && outputs[1].dim() == 1 && inputs[3].dim() == 2 &&
                inputs[3].size(1) == hidden, "group exchange value and scalar dimensions differ");
  for (size_t index = 0; index < tables.size(); ++index) {
    Tensor(tables[index], index == 3 ? at::kLong : at::kInt, device);
  }
  TORCH_CHECK(tables[0].dim() == 1 && tables[1].dim() == 2 && tables[2].dim() == 2 &&
                (expand ? tables[1].size(0) == tables[0].numel() : tables[0].numel() == 0) &&
                tables[1].size(1) >= 1 &&
                tables[1].size(1) == tables[2].size(1) && tables[3].sizes() == at::IntArrayRef({ranks, 8}) &&
                tables[4].dim() == 2 && tables[4].size(1) == 2,
              "group exchange source/target member maps or peer controls are malformed");
  const int64_t top_k = tables[1].size(1);
  const int64_t vector_bytes = !expand || value_type == at::kFloat ? 4 : 2;
  const int64_t scalar_bytes = phase == 1 || phase == 4 ? (top_k + 7) / 8 * 32 * quantum : 0;
  TORCH_CHECK(top_k < (1LL << 30) && quantum * hidden * vector_bytes + 63 + scalar_bytes <= rows * hidden * 2,
              "group packet exceeds its admitted symmetric window");
  TORCH_CHECK((!expand && outputs[0].size(0) == tables[2].size(0)) ||
                (expand && outputs[1].numel() == outputs[0].size(0)), "group output extent differs from its phase");
  if (phase == 3) {
    TORCH_CHECK(inputs[2].numel() == outputs[0].size(0) && inputs[3].sizes() == outputs[0].sizes(),
                "backward dispatch must retain independent expert V and one probability per selection");
  }
  TORCH_CHECK(tables[4].size(0) == (expand ? outputs[0].size(0) : inputs[0].size(0)),
              "expert row readiness does not cover the current input/output selections");
  Tensor(ring, at::kBFloat16, device);
  Tensor(signals, at::kInt, device);
  Tensor(headers, at::kLong, device);
  Tensor(progress, at::kLong, device);
  TORCH_CHECK(ring.sizes() == at::IntArrayRef({ranks, lanes, rows, hidden}) &&
                signals.sizes() == at::IntArrayRef({2, ranks, lanes, 16}) && headers.dim() == 3 &&
                headers.size(0) == ranks && headers.size(1) == lanes && headers.size(2) >= 8 &&
                progress.sizes() == at::IntArrayRef({ranks, lanes, 16}), "group workspace geometry differs from its manifest");
  TORCH_CHECK(ready.has_value() == expected.has_value(), "group producer flags need their expectations");
  if (ready) {
    Tensor(*ready, at::kInt, device);
    Tensor(*expected, at::kInt, device);
    TORCH_CHECK(ready->dim() == 3 && ready->size(1) == config[10] && ready->size(2) == 16 &&
                  expected->sizes() == ready->sizes().slice(0, 2), "group producer readiness has invalid dimensions");
  }
  if (published) {
    Tensor(*published, at::kInt, device);
    TORCH_CHECK(published->sizes() == at::IntArrayRef({config[11], config[10], 16}),
                "group receive readiness has invalid tile dimensions");
  }
}

void Run(at::TensorList inputs, at::TensorList outputs, at::TensorList tables,
         const at::Tensor &ring, const at::Tensor &signals, const at::Tensor &headers,
         const at::Tensor &progress, at::IntArrayRef config,
         const c10::optional<at::Tensor> &ready, const c10::optional<at::Tensor> &expected,
         const c10::optional<at::Tensor> &published) {
  Validate(inputs, outputs, tables, ring, signals, headers, progress, config, ready, expected, published);
  TORCH_CHECK(inputs[0].device().type() == c10::DeviceType::PrivateUse1, "group exchange requires an NPU");
  const c10_npu::NPUGuard guard(inputs[0].device());
  const auto stream = c10_npu::getCurrentNPUStream(inputs[0].device().index());
  std::vector<at::Tensor> retained(inputs.begin(), inputs.end());
  retained.insert(retained.end(), tables.begin(), tables.end());
  std::vector<at::Tensor> writers(outputs.begin(), outputs.end());
  writers.insert(writers.end(), {ring, signals, headers, progress});
  if (ready) {
    retained.insert(retained.end(), {*ready, *expected});
  }
  if (published) {
    writers.push_back(*published);
  }
  for (size_t index = 0; index < writers.size(); ++index) {
    for (const auto &input : retained) {
      at::assert_no_overlap(writers[index], input);
    }
    for (size_t previous = 0; previous < index; ++previous) {
      at::assert_no_overlap(writers[index], writers[previous]);
    }
  }
  retained.insert(retained.end(), writers.begin(), writers.end());
  for (const auto &value : retained) {
    TORCH_CHECK(value.storage_offset() >= 0 && value.storage().nbytes() / value.element_size() >=
                  static_cast<uint64_t>(value.storage_offset() + value.numel()), "group operand storage was invalidated");
    if (value.numel()) {
      c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
    }
  }
  const auto pointer = [](const at::Tensor &value) { return reinterpret_cast<uint64_t>(value.data_ptr()); };
  const auto optional = [&pointer](const c10::optional<at::Tensor> &value) {
    return value ? pointer(*value) : static_cast<uint64_t>(0);
  };
  routed::GroupLaunch launch{pointer(inputs[0]), pointer(outputs[0]), pointer(inputs[1]), pointer(inputs[2]),
    pointer(inputs[3]), pointer(outputs[1]), pointer(tables[0]), pointer(tables[1]), pointer(tables[2]),
    pointer(tables[3]), pointer(tables[4]), optional(ready), optional(expected), optional(published),
    pointer(ring), pointer(signals), pointer(headers), pointer(progress),
    inputs[0].size(0), outputs[0].size(0), inputs[1].numel(), outputs[1].numel(), inputs[3].size(0),
    tables[1].size(0), tables[2].size(0), config[6], static_cast<uint64_t>(config[9]),
    static_cast<int32_t>(config[1]), static_cast<int32_t>(config[2]), static_cast<int32_t>(config[3]),
    static_cast<int32_t>(config[4]), static_cast<int32_t>(headers.size(2)), static_cast<int32_t>(inputs[0].size(1)),
    static_cast<int32_t>(tables[1].size(1)), static_cast<int32_t>(config[5]), static_cast<int32_t>(config[0]),
    inputs[0].scalar_type() == at::kFloat, static_cast<int32_t>(config[7]), ready ? static_cast<int32_t>(ready->size(0)) : 0,
    static_cast<int32_t>(config[10]), static_cast<int32_t>(config[11])};
  const int32_t workers = config[8];
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsGroupExchange", [=]() -> int {
    (void)retained;
    return RoutedExpertsGroupExchange(stream.stream(false), &launch, workers);
  });
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_group_exchange(Tensor[] inputs, Tensor(a!)[] outputs, Tensor[] tables, Tensor(b!) ring, "
        "Tensor(c!) signals, Tensor(d!) headers, Tensor(e!) progress, int[] config, "
        "Tensor? ready=None, Tensor? expected=None, Tensor(f!)? published=None) -> ()");
  m.def("routed_experts_group_exchange_validate(Tensor[] inputs, Tensor[] outputs, Tensor[] tables, Tensor ring, "
        "Tensor signals, Tensor headers, Tensor progress, int[] config, "
        "Tensor? ready=None, Tensor? expected=None, Tensor? published=None) -> ()");
}
TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_group_exchange", &Run);
  m.impl("routed_experts_group_exchange_validate", &Validate);
}
TORCH_LIBRARY_IMPL(hyper_parallel, Meta, m) {
  m.impl("routed_experts_group_exchange", &Validate);
  m.impl("routed_experts_group_exchange_validate", &Validate);
}
