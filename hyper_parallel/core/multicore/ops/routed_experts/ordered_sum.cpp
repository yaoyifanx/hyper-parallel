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
#include "ordered_sum.h"

using namespace AscendC;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};
namespace routed = hyper_parallel::multicore::routed_experts;

namespace {

class OrderedSumWorker {
 public:
  __aicore__ inline void Init(GM_ADDR metadata, int64_t groups, int64_t tiles) {
    table_ = reinterpret_cast<const __gm__ int64_t *>(metadata);
    groups_ = groups;
    tiles_ = tiles;
    pipe_.InitBuffer(accumulator_, routed::kOrderedSumTile * sizeof(float));
    pipe_.InitBuffer(incoming_, routed::kOrderedSumTile * sizeof(float));
    pipe_.InitBuffer(output_, routed::kOrderedSumTile * sizeof(bfloat16_t));
  }

  __aicore__ inline void Run() {
    int64_t group = 0;
    for (int64_t tile = GetBlockIdx(); tile < tiles_; tile += GetBlockNum()) {
      while (group + 1 < groups_ && tile >= table_[(group + 1) * routed::kOrderedSumFields + 5]) {
        ++group;
      }
      const auto *row = table_ + group * routed::kOrderedSumFields;
      const int64_t start = (tile - row[5]) * routed::kOrderedSumTile;
      const uint32_t count = row[1] - start < routed::kOrderedSumTile ? row[1] - start : routed::kOrderedSumTile;
      auto accumulator = accumulator_.Get<float>();
      const auto *sources = table_ + row[2];
      if (row[4]) {
        Load(accumulator, sources[0], start, count);
      } else {
        Duplicate(accumulator, 0.0F, count);
        PipeBarrier<PIPE_V>();
      }
      for (int64_t source = row[4]; source < row[3]; ++source) {
        auto incoming = incoming_.Get<float>();
        Load(incoming, sources[source], start, count);
        Add(accumulator, accumulator, incoming, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      }
      Store(row[0], start, count);
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
  }

 private:
  __aicore__ inline void Load(LocalTensor<float> destination, int64_t address,
                              int64_t start, uint32_t count) {
    GlobalTensor<float> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address) + start, count);
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    const DataCopyPadExtParams<float> padding{false, 0, 0, 0.0F};
    DataCopyPad(destination, source, copy, padding);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }

  __aicore__ inline void Store(int64_t address, int64_t start, uint32_t count) {
    auto output = output_.Get<bfloat16_t>();
    Cast(output, accumulator_.Get<float>(), RoundMode::CAST_RINT, count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<bfloat16_t> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(address) + start, count);
    const DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(bfloat16_t)), 0, 0, 0};
    DataCopyPad(destination, output, copy);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }

  TPipe pipe_;
  TBuf<TPosition::VECCALC> accumulator_, incoming_, output_;
  const __gm__ int64_t *table_;
  int64_t groups_, tiles_;
};

__global__ __aicore__ void OrderedSumKernel(GM_ADDR metadata, int64_t groups, int64_t tiles) {
  OrderedSumWorker worker;
  worker.Init(metadata, groups, tiles);
  worker.Run();
}

}  // namespace

extern "C" int RoutedExpertsOrderedSum(void *stream, void *metadata, int64_t groups,
                                       int64_t tiles, int32_t cores) {
  if (groups < 1 || tiles < 1 || cores < 1 || cores > 48 || metadata == nullptr) {
    return -1;
  }
  const auto error = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  if (error != ACL_SUCCESS) {
    return static_cast<int>(error);
  }
  OrderedSumKernel<<<cores, nullptr, stream>>>(static_cast<uint8_t *>(metadata), groups, tiles);
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
