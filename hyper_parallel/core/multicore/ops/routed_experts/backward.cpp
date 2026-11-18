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

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "acl/acl.h"
#include "backward.h"
#include "gradient.h"
#include "progress.h"

using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};
namespace gradients = hyper_parallel::multicore::routed_experts;

namespace {

__aicore__ inline void LoadTiling(TCubeTiling &tiling, GM_ADDR source) {
  auto *local = reinterpret_cast<int32_t *>(&tiling);
  const auto *global = reinterpret_cast<const __gm__ int32_t *>(source);
  for (int32_t index = 0; index < sizeof(TCubeTiling) / sizeof(int32_t); ++index) {
    local[index] = global[index];
  }
}

template <bool TransposeB>
__aicore__ inline void InputGradient(GM_ADDR dz, GM_ADDR weights, GM_ADDR tasks, GM_ADDR output,
                                    GM_ADDR config, GM_ADDR ready, int32_t count,
                                    int32_t hidden, int32_t inner, int32_t experts) {
  using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
  using B = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, TransposeB>;
  using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
  TPipe pipe;
  matmul::Matmul<A, B, C, C, CFG_MDL> mm;
  TCubeTiling tiling;
  LoadTiling(tiling, config);
  REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &tiling);
  const auto *table = reinterpret_cast<const __gm__ int64_t *>(tasks);
  const auto *pointers = reinterpret_cast<const __gm__ int64_t *>(weights);
  int32_t previous = -1;
  routed::TaskProgressPublisher publisher;
  for (int32_t index = GetBlockIdx(); index < count; index += GetBlockNum()) {
    const auto *task = table + static_cast<int64_t>(index) * 5;
    if (task[0] < previous || task[0] < 0 || task[0] >= experts) {
      Trap();
    }
    if (task[0] != previous) {
      publisher.Finish();
      publisher.Init(ready, experts, task[0]);
      previous = task[0];
    }
    GlobalTensor<bfloat16_t> x, weight, result;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dz) + task[1] * inner * 2);
    weight.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pointers[task[0]]) +
                          (TransposeB ? task[3] * inner * 2 : task[3]));
    result.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output) + task[1] * hidden + task[3]);
    result.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    mm.SetOrgShape(task[2], hidden, inner * 2);
    mm.SetSingleShape(task[2], task[4], inner * 2);
    mm.SetTensorA(x);
    mm.SetTensorB(weight, TransposeB);
    mm.IterateAll(result);
    publisher.Publish(index);
  }
  publisher.Finish();
  mm.End();
}

__aicore__ inline void WeightGradient(GM_ADDR input, GM_ADDR derivative, GM_ADDR tasks,
                                     GM_ADDR output, GM_ADDR config, int32_t count,
                                     int32_t features, int32_t columns, int32_t experts,
                                     GM_ADDR early_experts, bool early) {
  using A = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
  using B = matmul::MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;
  using C = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
  TPipe pipe;
  matmul::Matmul<A, B, C, C, CFG_MDL> mm;
  TCubeTiling tiling;
  LoadTiling(tiling, config + sizeof(gradients::GradientConfigHeader));
  REGIST_MATMUL_OBJ(&pipe, nullptr, mm, &tiling);
  const auto *table = reinterpret_cast<const __gm__ int64_t *>(tasks);
  const auto *priority = reinterpret_cast<const __gm__ int32_t *>(early_experts);
  for (int32_t index = GetBlockIdx(); index < count; index += GetBlockNum()) {
    const auto *task = table + static_cast<int64_t>(index) * gradients::kGradientTaskFields;
    if (task[0] < 0 || task[0] >= experts || task[1] < 0 || task[1] + task[2] > features ||
        task[3] < 0 || task[3] + task[4] > columns || task[5] < 0 || task[6] < 1) {
      Trap();
    }
    if ((priority[task[0]] != 0) != early) {
      continue;
    }
    GlobalTensor<bfloat16_t> x, dy;
    GlobalTensor<float> dw;
    x.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(input) + task[5] * features + task[1]);
    dy.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(derivative) + task[5] * columns + task[3]);
    dw.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output) + (task[0] * features + task[1]) * columns + task[3]);
    dw.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    mm.SetOrgShape(features, columns, task[6]);
    mm.SetSingleShape(task[2], task[4], task[6]);
    mm.SetTensorA(x, true);
    mm.SetTensorB(dy, false);
    mm.IterateAll(dw, 0);
  }
  mm.End();
}

__aicore__ inline void PublishGradient(GM_ADDR flags) {
  auto *cell = reinterpret_cast<__gm__ int32_t *>(flags) + GetBlockIdx() * 16;
  GlobalTensor<int32_t> line;
  line.SetGlobalBuffer(cell, 16);
  PipeBarrier<PIPE_ALL>();
  __asm__ __volatile__("" ::: "memory");
  *reinterpret_cast<__gm__ volatile int32_t *>(cell) = 1;
  DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(line);
  __asm__ __volatile__("" ::: "memory");
}

template <bool TransposeB>
__global__ __aicore__ void BackwardTailKernel(GM_ADDR dz, GM_ADDR weights, GM_ADDR dx_tasks,
    GM_ADDR dx, GM_ADDR dx_config, GM_ADDR ready, GM_ADDR x, GM_ADDR activation, GM_ADDR dv,
    GM_ADDR gate_tasks, GM_ADDR gate_gradient, GM_ADDR gate_config,
    GM_ADDR down_tasks, GM_ADDR down_gradient, GM_ADDR down_config, GM_ADDR early_experts, GM_ADDR gradient_ready,
    int32_t dx_count, int32_t gate_count, int32_t down_count,
    int32_t hidden, int32_t inner, int32_t experts) {
  WeightGradient(x, dz, gate_tasks, gate_gradient, gate_config, gate_count, hidden, inner * 2, experts,
                 early_experts, true);
  WeightGradient(activation, dv, down_tasks, down_gradient, down_config, down_count, inner, hidden, experts,
                 early_experts, true);
  PublishGradient(gradient_ready);
  InputGradient<TransposeB>(dz, weights, dx_tasks, dx, dx_config, ready, dx_count, hidden, inner, experts);
  // Cores retain their unique complete-token output tiles. There is no global
  // dX barrier: a core can produce dW while transport consumes published dX.
  WeightGradient(x, dz, gate_tasks, gate_gradient, gate_config, gate_count, hidden, inner * 2, experts,
                 early_experts, false);
  WeightGradient(activation, dv, down_tasks, down_gradient, down_config, down_count, inner, hidden, experts,
                 early_experts, false);
}

}  // namespace

extern "C" int RoutedExpertsBackwardTail(void *stream, void *dz, void *weights, void *dx_tasks,
    void *dx, void *dx_config, void *output_ready, void *x, void *activation, void *dv,
    void *gate_tasks, void *gate_gradient, void *gate_config,
    void *down_tasks, void *down_gradient, void *down_config, void *early_experts, void *gradient_ready,
    int32_t dx_count, int32_t gate_count, int32_t down_count,
    int32_t hidden, int32_t inner, int32_t experts, int32_t cores, bool transpose_b) {
  if (cores < 1 || cores > 24 || dx_count < cores || gate_count < 1 || down_count < 1 ||
      hidden < 16 || hidden > 32768 || hidden % 16 || inner < 16 || inner > 32768 || inner % 16 || experts < 1) {
    return -1;
  }
  const auto error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (error != ACL_SUCCESS) {
    return static_cast<int>(error);
  }
#define HP_BACKWARD_LAUNCH(transpose) \
  BackwardTailKernel<transpose><<<cores, nullptr, stream>>>( \
    static_cast<uint8_t *>(dz), static_cast<uint8_t *>(weights), static_cast<uint8_t *>(dx_tasks), \
    static_cast<uint8_t *>(dx), static_cast<uint8_t *>(dx_config), static_cast<uint8_t *>(output_ready), \
    static_cast<uint8_t *>(x), static_cast<uint8_t *>(activation), static_cast<uint8_t *>(dv), \
    static_cast<uint8_t *>(gate_tasks), static_cast<uint8_t *>(gate_gradient), static_cast<uint8_t *>(gate_config), \
    static_cast<uint8_t *>(down_tasks), static_cast<uint8_t *>(down_gradient), static_cast<uint8_t *>(down_config), \
    static_cast<uint8_t *>(early_experts), static_cast<uint8_t *>(gradient_ready), \
    dx_count, gate_count, down_count, hidden, inner, experts)
  if (transpose_b) {
    HP_BACKWARD_LAUNCH(true);
  } else {
    HP_BACKWARD_LAUNCH(false);
  }
#undef HP_BACKWARD_LAUNCH
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
