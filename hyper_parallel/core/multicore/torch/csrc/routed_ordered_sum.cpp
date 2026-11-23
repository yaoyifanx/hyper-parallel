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
#include <limits>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "ordered_sum.h"

namespace routed = hyper_parallel::multicore::routed_experts;
namespace {

void CheckTensor(const at::Tensor &value, c10::Device device, at::ScalarType dtype) {
  TORCH_CHECK(value.device() == device && value.scalar_type() == dtype && value.is_contiguous(),
              "ordered sum requires contiguous operands on one NPU with matching dtypes");
  TORCH_CHECK(value.storage_offset() >= 0 && value.storage().nbytes() / value.element_size() >=
                static_cast<uint64_t>(value.storage_offset() + value.numel()),
              "ordered sum operand has invalidated storage");
}

int64_t CheckOutputs(at::TensorList sources, at::TensorList outputs) {
  TORCH_CHECK(!outputs.empty(), "ordered sum requires at least one output");
  const auto device = outputs.front().device();
  TORCH_CHECK(device.type() == c10::DeviceType::PrivateUse1, "ordered sum requires NPU outputs");
  for (const auto &source : sources) {
    CheckTensor(source, device, at::kFloat);
  }
  int64_t tiles = 0;
  for (size_t index = 0; index < outputs.size(); ++index) {
    const auto &output = outputs[index];
    CheckTensor(output, device, at::kBFloat16);
    const auto count = (output.numel() + routed::kOrderedSumTile - 1) / routed::kOrderedSumTile;
    TORCH_CHECK(count <= INT64_MAX - tiles, "ordered sum tile count overflows int64");
    tiles += count;
    for (const auto &source : sources) {
      at::assert_no_overlap(output, source);
    }
    for (size_t previous = 0; previous < index; ++previous) {
      at::assert_no_overlap(output, outputs[previous]);
    }
  }
  return tiles;
}

at::Tensor Prepare(at::TensorList sources, at::TensorList outputs,
                    at::IntArrayRef offsets, at::IntArrayRef local) {
  CheckOutputs(sources, outputs);
  TORCH_CHECK(offsets.size() == outputs.size() + 1 && local.size() == outputs.size() &&
                offsets.front() == 0 && offsets.back() == static_cast<int64_t>(sources.size()),
              "ordered sum offsets must partition the source list once per output");
  const c10_npu::NPUGuard guard(outputs.front().device());
  const int64_t groups = outputs.size();
  const int64_t fields = groups * routed::kOrderedSumFields;
  auto host = at::empty({fields + static_cast<int64_t>(sources.size())},
                       at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  auto *table = host.mutable_data_ptr<int64_t>();
  int64_t tile = 0;
  for (int64_t group = 0; group < groups; ++group) {
    TORCH_CHECK(offsets[group] >= 0 && offsets[group] <= offsets[group + 1] &&
                  offsets[group + 1] <= static_cast<int64_t>(sources.size()) &&
                  (local[group] == 0 || local[group] == 1) && local[group] <= offsets[group + 1] - offsets[group],
                "ordered sum has malformed source groups or a missing local contribution");
    auto *row = table + group * routed::kOrderedSumFields;
    row[0] = reinterpret_cast<int64_t>(outputs[group].data_ptr());
    row[1] = outputs[group].numel();
    row[2] = fields + offsets[group];
    row[3] = offsets[group + 1] - offsets[group];
    row[4] = local[group];
    row[5] = tile;
    tile += (outputs[group].numel() + routed::kOrderedSumTile - 1) / routed::kOrderedSumTile;
    for (int64_t index = offsets[group]; index < offsets[group + 1]; ++index) {
      TORCH_CHECK(sources[index].numel() == outputs[group].numel(), "ordered sum source extent differs from output");
      table[fields + index] = reinterpret_cast<int64_t>(sources[index].data_ptr());
    }
  }
  return host.to(outputs.front().device(), at::kLong, true, false);
}

void Run(at::TensorList sources, at::TensorList outputs, const at::Tensor &metadata) {
  const auto tiles = CheckOutputs(sources, outputs);
  const auto device = outputs.front().device();
  CheckTensor(metadata, device, at::kLong);
  TORCH_CHECK(metadata.dim() == 1 && metadata.numel() ==
                static_cast<int64_t>(outputs.size()) * routed::kOrderedSumFields + static_cast<int64_t>(sources.size()),
              "ordered sum metadata has the wrong extent");
  const c10_npu::NPUGuard guard(device);
  const auto stream = c10_npu::getCurrentNPUStream(device.index());
  std::vector<at::Tensor> retained(sources.begin(), sources.end());
  retained.insert(retained.end(), outputs.begin(), outputs.end());
  retained.push_back(metadata);
  for (const auto &value : retained) {
    c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
  }
  if (!tiles) {
    return;
  }
  const int64_t groups = outputs.size();
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsOrderedSum", [=]() -> int {
    (void)retained;
    return RoutedExpertsOrderedSum(stream.stream(false), metadata.data_ptr(), groups, tiles,
                                    static_cast<int32_t>(std::min<int64_t>(48, tiles)));
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_ordered_sum_prepare(Tensor[] sources, Tensor[] outputs, int[] offsets, int[] local) -> Tensor");
  m.def("routed_experts_ordered_sum(Tensor[] sources, Tensor(a!)[] outputs, Tensor metadata) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_ordered_sum_prepare", &Prepare);
  m.impl("routed_experts_ordered_sum", &Run);
}
