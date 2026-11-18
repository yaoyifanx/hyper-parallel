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
#include "acl/acl.h"
#include "backward.h"

using namespace AscendC;

namespace {

__global__ __aicore__ void WaitProducersKernel(GM_ADDR flags, int32_t producers, uint64_t watchdog) {
  auto *cells = reinterpret_cast<__gm__ int32_t *>(flags);
  const auto started = GetSystemCycle();
  for (int32_t index = 0; index < producers; ++index) {
    GlobalTensor<int32_t> line;
    line.SetGlobalBuffer(cells + index * 16, 16);
    while (true) {
      __asm__ __volatile__("" ::: "memory");
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(line);
      __asm__ __volatile__("" ::: "memory");
      if (*reinterpret_cast<__gm__ volatile int32_t *>(cells + index * 16) == 1) {
        break;
      }
      if (GetSystemCycle() - started >= watchdog) {
        Trap();
      }
    }
  }
  PipeBarrier<PIPE_ALL>();
}

}  // namespace

extern "C" int RoutedExpertsWaitProducers(void *stream, void *flags, int32_t producers, int64_t watchdog) {
  if (producers < 0 || producers > 24 || watchdog < 1 ||
      (producers && (flags == nullptr || reinterpret_cast<uintptr_t>(flags) % 64))) {
    return -1;
  }
  if (producers) {
    WaitProducersKernel<<<1, nullptr, stream>>>(static_cast<uint8_t *>(flags), producers, watchdog);
  }
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
