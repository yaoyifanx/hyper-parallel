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
#include <torch/library.h>
#include <vector>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "backward.h"
#include "gradient.h"
#include "transport.h"

namespace {

void CheckStorage(const at::Tensor &value, const at::Tensor &anchor, at::ScalarType dtype) {
  TORCH_CHECK(value.device() == anchor.device() && value.is_contiguous() && value.scalar_type() == dtype,
              "backward tail operands must be contiguous with the expected device and dtype");
  if (!value.numel()) {
    return;
  }
  TORCH_CHECK(value.storage_offset() >= 0 && value.storage().nbytes() / value.element_size() >=
                static_cast<uint64_t>(value.storage_offset() + value.numel()) &&
                reinterpret_cast<uintptr_t>(value.data_ptr()) % 32 == 0,
              "backward tail operand has invalid or unaligned storage");
}

void CheckTasks(const at::Tensor &tasks, int64_t fields) {
  TORCH_CHECK(tasks.dim() == 2 && tasks.size(1) == fields && tasks.size(0) > 0 && tasks.size(0) <= INT32_MAX,
              "backward tail tasks must contain nonempty canonical descriptors");
}

void CheckAliases(const std::vector<at::Tensor> &values, size_t readers,
                  size_t consumed_writer = SIZE_MAX, bool disjoint_storage = false) {
  for (size_t writer = readers; writer < values.size(); ++writer) {
    for (size_t index = 0; index < values.size(); ++index) {
      if (index == writer || !values[writer].numel() || !values[index].numel()) {
        continue;
      }
      if (writer == consumed_writer && index == 6) { continue; }
      const auto start = reinterpret_cast<uintptr_t>(values[writer].data_ptr());
      const auto other = reinterpret_cast<uintptr_t>(values[index].data_ptr());
      const auto bytes = values[writer].numel() * values[writer].element_size();
      const auto other_bytes = values[index].numel() * values[index].element_size();
      TORCH_CHECK((disjoint_storage || !values[writer].is_alias_of(values[index])) &&
                    (start <= other ? other - start >= bytes : start - other >= other_bytes),
                  "backward tail writable storage overlaps another operand");
    }
  }
}

void CheckMatrixShapes(const std::vector<at::Tensor> &inputs, const at::Tensor &dx,
                       const at::Tensor &gate, const at::Tensor &down) {
  const auto &dz = inputs[0];
  const auto &x = inputs[4];
  const auto &activation = inputs[5];
  const auto &dv = inputs[6];
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && x.dim() == 2 && x.size(0) > 0 &&
                x.size(0) <= (1 << 20) && x.size(1) >= 16 && x.size(1) <= 32768 && x.size(1) % 16 == 0 &&
                dz.dim() == 2 && dz.size(0) == x.size(0) && dz.size(1) >= 32 && dz.size(1) <= 65536 &&
                dz.size(1) % 32 == 0 && dv.sizes() == x.sizes() && dx.sizes() == x.sizes(),
              "backward tail matrix shapes are invalid");
  const int64_t inner = dz.size(1) / 2;
  TORCH_CHECK(activation.sizes() == at::IntArrayRef({x.size(0), inner}) &&
                gate.dim() == 3 && gate.size(0) > 0 && gate.size(1) == x.size(1) && gate.size(2) == inner * 2 &&
                down.sizes() == at::IntArrayRef({gate.size(0), inner, x.size(1)}),
              "backward tail gradient or activation shapes are invalid");
}

void CheckControls(const std::vector<at::Tensor> &inputs, const at::Tensor &gate,
                    const at::Tensor &ready, int64_t cores) {
  TORCH_CHECK(cores >= 1 && cores <= 24 && inputs[2].size(0) >= cores &&
                ready.sizes() == at::IntArrayRef({cores, gate.size(0), 16}) &&
                reinterpret_cast<uintptr_t>(ready.data_ptr()) % 64 == 0,
              "backward tail readiness must cover every cube producer and actual expert");
  TORCH_CHECK(inputs[1].sizes() == at::IntArrayRef({gate.size(0)}) &&
                inputs[3].numel() == static_cast<int64_t>(RoutedExpertsGemmConfigBytes()) &&
                inputs[8].numel() == static_cast<int64_t>(RoutedExpertsGradientConfigBytes()) &&
                inputs[10].numel() == static_cast<int64_t>(RoutedExpertsGradientConfigBytes()),
              "backward tail pointer table or tiling extent is invalid");
}

void RunBackwardTail(at::TensorList weights, at::TensorList operands, at::Tensor dx,
                  at::Tensor gate_gradient, at::Tensor down_gradient, at::Tensor ready,
                  at::Tensor gradient_ready, bool transposed, int64_t cores,
                  bool down_complete, bool consume_derivative) {
  TORCH_CHECK(operands.size() == 12, "backward tail expects twelve prepared read-only operands");
  std::vector<at::Tensor> inputs(operands.begin(), operands.end());
  const auto &anchor = inputs[0];
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto dtype = index == 11 ? at::kInt : (index == 1 || index == 2 || index == 7 || index == 9) ? at::kLong :
                       (index == 3 || index == 8 || index == 10) ? at::kByte : at::kBFloat16;
    CheckStorage(inputs[index], anchor, dtype);
  }
  CheckStorage(dx, anchor, at::kBFloat16);
  CheckStorage(gate_gradient, anchor, at::kFloat);
  CheckStorage(down_gradient, anchor, at::kFloat);
  CheckStorage(ready, anchor, at::kInt);
  CheckStorage(gradient_ready, anchor, at::kInt);
  CheckTasks(inputs[2], 5);
  CheckTasks(inputs[7], 7);
  CheckTasks(inputs[9], 7);
  CheckMatrixShapes(inputs, dx, gate_gradient, down_gradient);
  CheckControls(inputs, gate_gradient, ready, cores);
  TORCH_CHECK(inputs[11].sizes() == at::IntArrayRef({gate_gradient.size(0)}) &&
                gradient_ready.sizes() == at::IntArrayRef({cores, 16}) &&
                reinterpret_cast<uintptr_t>(gradient_ready.data_ptr()) % 64 == 0,
              "early expert priorities and gradient producer flags have invalid geometry");
  TORCH_CHECK(weights.size() == static_cast<size_t>(gate_gradient.size(0)), "backward weights miss actual experts");
  auto retained = inputs;
  for (const auto &weight : weights) {
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({anchor.size(1), dx.size(1)}), "invalid dX weight shape");
    CheckStorage(transposed ? weight.transpose(0, 1) : weight, anchor, at::kBFloat16);
    retained.push_back(weight);
  }
  const auto readers = retained.size();
  retained.insert(retained.end(), {dx, gate_gradient, down_gradient, ready, gradient_ready});
  TORCH_CHECK(!consume_derivative ||
                  (down_complete && dx.data_ptr() == inputs[6].data_ptr() && dx.sizes() == inputs[6].sizes() &&
                   dx.strides() == inputs[6].strides()), "consumed dY must be the exact dX output after W2");
  CheckAliases(retained, readers, consume_derivative ? readers : SIZE_MAX);
  const c10_npu::NPUGuard guard(anchor.device());
  const auto stream = c10_npu::getCurrentNPUStream(anchor.device().index());
  for (const auto &value : retained) {
    c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
  }
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsBackwardTail", [=]() -> int {
    // Retain references captured by the asynchronous command independently of TensorList's caller.
    (void)retained;
    const auto launch = down_complete ? RoutedExpertsBackwardTailW2First : RoutedExpertsBackwardTail;
    return launch(stream.stream(false), inputs[0].data_ptr(), inputs[1].data_ptr(),
      inputs[2].data_ptr(), dx.data_ptr(), inputs[3].data_ptr(), ready.data_ptr(),
      inputs[4].data_ptr(), inputs[5].data_ptr(), inputs[6].data_ptr(),
      inputs[7].data_ptr(), gate_gradient.data_ptr(), inputs[8].data_ptr(),
      inputs[9].data_ptr(), down_gradient.data_ptr(), inputs[10].data_ptr(), inputs[11].data_ptr(),
      gradient_ready.data_ptr(),
      inputs[2].size(0), inputs[7].size(0), inputs[9].size(0), dx.size(1), anchor.size(1) / 2,
      gate_gradient.size(0), cores, transposed);
  });
}

void RunBackwardAccumulate(at::TensorList weights, at::TensorList operands, at::Tensor dx,
                        at::TensorList gate_outputs, at::TensorList down_outputs, at::Tensor ready,
                        at::Tensor gradient_ready, bool transposed, bool gate_transposed,
                        bool down_transposed, int64_t cores, bool down_complete, bool consume_derivative) {
  TORCH_CHECK(operands.size() == 14 && !gate_outputs.empty() && gate_outputs.size() == down_outputs.size(),
              "mapped backward expects fourteen operands and gate/down destinations");
  std::vector<at::Tensor> inputs(operands.begin(), operands.end());
  const auto &anchor = inputs[0];
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto dtype = index == 11 ? at::kInt : (index == 1 || index == 2 || index == 7 || index == 9 || index >= 12)
        ? at::kLong : (index == 3 || index == 8 || index == 10) ? at::kByte : at::kBFloat16;
    CheckStorage(inputs[index], anchor, dtype);
  }
  CheckStorage(dx, anchor, at::kBFloat16);
  CheckStorage(ready, anchor, at::kInt);
  CheckStorage(gradient_ready, anchor, at::kInt);
  CheckTasks(inputs[2], 5);
  CheckTasks(inputs[7], 7);
  CheckTasks(inputs[9], 7);
  const int64_t experts = gate_outputs.size(), hidden = dx.size(1), inner = anchor.size(1) / 2;
  const auto meta = at::TensorOptions().device(at::kMeta).dtype(at::kFloat);
  const auto gate_shape = at::empty({experts, hidden, inner * 2}, meta);
  CheckMatrixShapes(inputs, dx, gate_shape, at::empty({experts, inner, hidden}, meta));
  CheckControls(inputs, gate_shape, ready, cores);
  TORCH_CHECK(weights.size() == gate_outputs.size() && inputs[11].numel() == experts &&
                  inputs[12].numel() == experts && inputs[13].numel() == experts &&
                  gradient_ready.sizes() == at::IntArrayRef({cores, 16}), "mapped backward metadata is invalid");
  auto retained = inputs;
  for (const auto &weight : weights) {
    TORCH_CHECK(weight.sizes() == at::IntArrayRef({inner * 2, hidden}), "invalid mapped dX weight");
    CheckStorage(transposed ? weight.transpose(0, 1) : weight, anchor, at::kBFloat16);
    retained.push_back(weight);
  }
  const auto readers = retained.size();
  for (size_t role = 0; role < 2; ++role) {
    const auto outputs = role ? down_outputs : gate_outputs;
    const auto transpose = role ? down_transposed : gate_transposed;
    const int64_t rows = role ? inner : hidden, columns = role ? hidden : inner * 2;
    for (const auto &output : outputs) {
      CheckStorage(output, anchor, at::kFloat);
      TORCH_CHECK(output.sizes() == at::IntArrayRef({transpose ? columns : rows, transpose ? rows : columns}),
                  "mapped dW must use contiguous physical parameter layout");
      retained.push_back(output);
    }
  }
  const auto dx_index = retained.size();
  retained.insert(retained.end(), {dx, ready, gradient_ready});
  TORCH_CHECK(!consume_derivative ||
                  (down_complete && dx.data_ptr() == inputs[6].data_ptr() && dx.sizes() == inputs[6].sizes() &&
                   dx.strides() == inputs[6].strides()), "consumed dY must be the exact dX output after W2");
  CheckAliases(retained, readers, consume_derivative ? dx_index : SIZE_MAX, true);
  const c10_npu::NPUGuard guard(anchor.device());
  const auto stream = c10_npu::getCurrentNPUStream(anchor.device().index());
  for (const auto &value : retained) {
    c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
  }
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsBackwardAccumulate", [=]() -> int {
    (void)retained;
    const auto launch = down_complete ? RoutedExpertsBackwardAccumulateW2First : RoutedExpertsBackwardAccumulate;
    return launch(stream.stream(false), inputs[0].data_ptr(), inputs[1].data_ptr(),
      inputs[2].data_ptr(), dx.data_ptr(), inputs[3].data_ptr(), ready.data_ptr(),
      inputs[4].data_ptr(), inputs[5].data_ptr(), inputs[6].data_ptr(),
      inputs[7].data_ptr(), inputs[12].data_ptr(), inputs[8].data_ptr(),
      inputs[9].data_ptr(), inputs[13].data_ptr(), inputs[10].data_ptr(), inputs[11].data_ptr(),
      gradient_ready.data_ptr(), inputs[2].size(0), inputs[7].size(0), inputs[9].size(0),
      hidden, inner, experts, cores, transposed, gate_transposed, down_transposed);
  });
}

void BackwardTail(at::TensorList weights, at::TensorList operands, at::Tensor dx,
                  at::Tensor gate, at::Tensor down, at::Tensor ready, at::Tensor grad_ready,
                  bool transposed, int64_t cores) {
  RunBackwardTail(weights, operands, dx, gate, down, ready, grad_ready, transposed, cores, false, false);
}

void BackwardTailW2First(at::TensorList weights, at::TensorList operands, at::Tensor dx,
                        at::Tensor gate, at::Tensor down, at::Tensor ready, at::Tensor grad_ready,
                        bool transposed, int64_t cores, bool consume_derivative) {
  RunBackwardTail(weights, operands, dx, gate, down, ready, grad_ready, transposed, cores, true, consume_derivative);
}

void BackwardAccumulate(at::TensorList weights, at::TensorList operands, at::Tensor dx,
                        at::TensorList gate, at::TensorList down, at::Tensor ready, at::Tensor grad_ready,
                        bool transposed, bool gate_transposed, bool down_transposed, int64_t cores) {
  RunBackwardAccumulate(weights, operands, dx, gate, down, ready, grad_ready,
                        transposed, gate_transposed, down_transposed, cores, false, false);
}

void BackwardAccumulateW2First(at::TensorList weights, at::TensorList operands, at::Tensor dx,
                               at::TensorList gate, at::TensorList down, at::Tensor ready, at::Tensor grad_ready,
                               bool transposed, bool gate_transposed, bool down_transposed,
                               int64_t cores, bool consume_derivative) {
  RunBackwardAccumulate(weights, operands, dx, gate, down, ready, grad_ready,
                        transposed, gate_transposed, down_transposed, cores, true, consume_derivative);
}

void MappedGradient(at::TensorList operands, at::TensorList outputs, int64_t cores) {
  TORCH_CHECK(operands.size() == 6 && !outputs.empty(), "mapped gradient requires six operands and outputs");
  std::vector<at::Tensor> inputs(operands.begin(), operands.end());
  const auto &x = inputs[0];
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto dtype = index < 2 ? at::kBFloat16 : index == 3 ? at::kByte : index == 5 ? at::kInt : at::kLong;
    CheckStorage(inputs[index], x, dtype);
  }
  TORCH_CHECK(x.dim() == 2 && inputs[1].dim() == 2 && x.size(0) == inputs[1].size(0) && x.size(0) > 0 &&
                  cores > 0 && cores <= 24, "mapped gradient input shapes or core count are invalid");
  CheckTasks(inputs[2], 7);
  TORCH_CHECK(inputs[2].size(0) >= cores && inputs[3].numel() == RoutedExpertsGradientConfigBytes() &&
                  inputs[4].numel() == static_cast<int64_t>(outputs.size()) &&
                  inputs[5].numel() == static_cast<int64_t>(outputs.size()), "mapped gradient metadata is invalid");
  auto retained = inputs;
  for (const auto &output : outputs) {
    CheckStorage(output, x, at::kFloat);
    TORCH_CHECK(output.sizes() == at::IntArrayRef({x.size(1), inputs[1].size(1)}), "invalid mapped output shape");
    retained.push_back(output);
  }
  CheckAliases(retained, inputs.size(), SIZE_MAX, true);
  const c10_npu::NPUGuard guard(x.device());
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  for (const auto &value : retained) {
    c10_npu::NPUCachingAllocator::recordStream(value.storage().data_ptr(), stream);
  }
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsMappedGradient", [=]() -> int {
    (void)retained;
    return RoutedExpertsMappedGradient(stream.stream(false), x.data_ptr(), inputs[1].data_ptr(),
        inputs[2].data_ptr(), inputs[4].data_ptr(), inputs[3].data_ptr(), inputs[5].data_ptr(),
        inputs[2].size(0), x.size(1), inputs[1].size(1), outputs.size(), cores);
  });
}

void WaitProducers(const at::Tensor &flags, int64_t watchdog) {
  TORCH_CHECK(flags.device().type() == c10::DeviceType::PrivateUse1 && flags.dim() == 2 &&
                flags.size(0) <= 24 && flags.size(1) == 16 && watchdog > 0,
              "producer wait requires aligned per-core cache lines and a positive watchdog");
  CheckStorage(flags, flags, at::kInt);
  const c10_npu::NPUGuard guard(flags.device());
  const auto stream = c10_npu::getCurrentNPUStream(flags.device().index());
  c10_npu::NPUCachingAllocator::recordStream(flags.storage().data_ptr(), stream);
  at_npu::native::OpCommand::RunOpApi("RoutedExpertsWaitProducers", [=]() -> int {
    return RoutedExpertsWaitProducers(stream.stream(false), flags.data_ptr(), flags.size(0), watchdog);
  });
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(hyper_parallel, m) {
  m.def("routed_experts_backward_tail(Tensor[] weights, Tensor[] operands, Tensor(a!) dx, "
        "Tensor(b!) gate_gradient, Tensor(c!) down_gradient, Tensor(d!) ready, Tensor(e!) gradient_ready, "
        "bool transposed, int cores) -> ()");
  m.def("routed_experts_backward_accumulate(Tensor[] weights, Tensor[] operands, Tensor(a!) dx, "
        "Tensor(b!)[] gate_outputs, Tensor(c!)[] down_outputs, Tensor(d!) ready, Tensor(e!) gradient_ready, "
        "bool transposed, bool gate_transposed, bool down_transposed, int cores) -> ()");
  m.def("routed_experts_backward_tail_w2_first(Tensor[] weights, Tensor[] operands, Tensor(a!) dx, "
        "Tensor(b!) gate_gradient, Tensor(c!) down_gradient, Tensor(d!) ready, Tensor(e!) gradient_ready, "
        "bool transposed, int cores, bool consume_derivative=False) -> ()");
  m.def("routed_experts_backward_accumulate_w2_first(Tensor[] weights, Tensor[] operands, Tensor(a!) dx, "
        "Tensor(b!)[] gate_outputs, Tensor(c!)[] down_outputs, Tensor(d!) ready, Tensor(e!) gradient_ready, "
        "bool transposed, bool gate_transposed, bool down_transposed, int cores, "
        "bool consume_derivative=False) -> ()");
  m.def("routed_experts_mapped_gradient(Tensor[] operands, Tensor(a!)[] outputs, int cores) -> ()");
  m.def("routed_experts_wait_producers(Tensor flags, int watchdog=60000000000) -> ()");
}

TORCH_LIBRARY_IMPL(hyper_parallel, PrivateUse1, m) {
  m.impl("routed_experts_backward_tail", &BackwardTail);
  m.impl("routed_experts_backward_accumulate", &BackwardAccumulate);
  m.impl("routed_experts_backward_tail_w2_first", &BackwardTailW2First);
  m.impl("routed_experts_backward_accumulate_w2_first", &BackwardAccumulateW2First);
  m.impl("routed_experts_mapped_gradient", &MappedGradient);
  m.impl("routed_experts_wait_producers", &WaitProducers);
}
