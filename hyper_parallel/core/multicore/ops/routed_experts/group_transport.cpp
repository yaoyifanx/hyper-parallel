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
#include "data_plane/mte_rma.h"
#include "data_plane/sync.h"
#include "group_transport.h"
#include "progress.h"
#include "transport.h"

using namespace AscendC;
namespace communication = hyper_parallel::multicore::shmem::data_plane;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg {0};

namespace {
constexpr uint32_t kTile = 4096;
constexpr int64_t kMagic = 0x4d4f4f4e47525032LL;

class GroupWorker {
 public:
  __aicore__ inline void Init(routed::GroupLaunch args) {
    a_ = args;
    input_ = reinterpret_cast<GM_ADDR>(a_.input);
    output_ = reinterpret_cast<GM_ADDR>(a_.output);
    source_tokens_ = reinterpret_cast<__gm__ int32_t *>(a_.source_tokens);
    source_members_ = reinterpret_cast<__gm__ int32_t *>(a_.source_members);
    target_members_ = reinterpret_cast<__gm__ int32_t *>(a_.target_members);
    geometry_ = reinterpret_cast<__gm__ int64_t *>(a_.peer_offsets);
    progress_ = reinterpret_cast<__gm__ int64_t *>(a_.progress);
    // BD has no producer-task cache when G is supplied directly. Reuse that
    // UB budget for larger copies; dot reductions retain their original 4K tiles.
    backward_tile_ = a_.phase == 3 && a_.top_k <= 256 && a_.input_ready == 0 ? kTile * 2 : kTile;
    const bool partial = a_.phase == 2 || a_.phase == 4;
    pipe_.InitBuffer(acc_, backward_tile_ * sizeof(float));
    pipe_.InitBuffer(pack_acc_, partial ? kTile * sizeof(float) : 32);
    pipe_.InitBuffer(work_, backward_tile_ * sizeof(float));
    pipe_.InitBuffer(other_, backward_tile_ * sizeof(float));
    pipe_.InitBuffer(small_, backward_tile_ * sizeof(bfloat16_t));
    pipe_.InitBuffer(pack_small_, partial ? kTile * sizeof(bfloat16_t) : 32);
    pipe_.InitBuffer(header_, 64);
    pipe_.InitBuffer(scalar_, 256 * sizeof(float));
    scalar_stride_ = (a_.top_k + 7) / 8 * 8;
    cache_packet_scalars_ = (a_.phase == 1 || a_.phase == 4) && a_.top_k <= 256 &&
      static_cast<int64_t>(scalar_stride_) * a_.quantum <= 2048;
    const uint32_t packet_values = cache_packet_scalars_ ? scalar_stride_ * a_.quantum :
      (a_.phase == 3 && a_.top_k <= 256 ? a_.top_k * 8 : 256);
    pipe_.InitBuffer(packet_scalars_, packet_values * sizeof(float));
    pipe_.InitBuffer(scalar_rows_, 16 * sizeof(float));
    pipe_.InitBuffer(atomic_, 32);
    pipe_.InitBuffer(ready_cache_, 32768);
    pipe_.InitBuffer(required_cache_, a_.input_ready == 0 ? 32768 : 65536);
    scalar_stride_ = (a_.top_k + 7) / 8 * 8;
    wire_float_ = a_.phase == 2 || a_.phase == 4 || a_.input_float;
  }

  __aicore__ inline void Run() {
    const int32_t stripes = GetBlockNum() == a_.ranks * a_.lanes ? a_.lanes : 1;
    const int32_t stripe = GetBlockIdx() % stripes;
    bool pending = true;
    while (pending) {
      pending = false;
      for (int32_t peer = GetBlockIdx() / stripes; peer < a_.ranks; peer += GetBlockNum() / stripes) {
        auto *state = progress_ + static_cast<int64_t>(peer * a_.lanes + stripe) * 16;
        const auto *geometry = geometry_ + static_cast<int64_t>(peer) * 8;
        const int64_t send_offset = StripeOffset(state[0], geometry[4], stripe, stripes);
        const int64_t receive_offset = StripeOffset(state[1], geometry[5], stripe, stripes);
        if (send_offset >= geometry[1] && receive_offset >= geometry[3]) {
          continue;
        }
        if (state[12] == 0) {
          if (peer != a_.rank && send_offset < geometry[1]) {
            uint64_t address = 0, capacity = 0;
            if (!hyper_parallel::multicore::shmem::cann::device::query_mte_resources(peer, address, capacity)) {
              state[4] = -2;
              Trap();
            }
          }
          state[2] = GetSystemCycle();
          state[12] = GetSystemCycle();
        }
        bool advanced = false;
        if (peer == a_.rank) {
          if (send_offset < geometry[1] &&
              ProbeInput(geometry[0] + send_offset, Groups(geometry[1] - send_offset), state)) {
            const int32_t count = Groups(geometry[1] - send_offset);
            auto begin = GetSystemCycle();
            Pack(Ring(peer, stripe), geometry[0] + send_offset, count);
            state[13] += GetSystemCycle() - begin;
            begin = GetSystemCycle();
            Unpack(Ring(peer, stripe), geometry[2] + receive_offset, count);
            state[14] += GetSystemCycle() - begin;
            state[0] += count;
            state[1] += count;
            state[11] += count;
            advanced = true;
          }
        } else {
          const int32_t receive_sequence = geometry[5] + receive_offset / a_.quantum + 1;
          const int32_t receive_lane = (receive_sequence - 1) % a_.lanes;
          if (receive_offset < geometry[3] && Test(0, peer, receive_lane, receive_sequence)) {
            const int32_t count = Groups(geometry[3] - receive_offset);
            state[8] += CheckHeader(Header(peer, receive_lane), receive_sequence, count, geometry[2] + receive_offset);
            const auto begin = GetSystemCycle();
            Unpack(Ring(peer, receive_lane), geometry[2] + receive_offset, count);
            state[14] += GetSystemCycle() - begin;
            communication::signal_set(Signal(1, a_.rank, receive_lane), receive_sequence + a_.lanes, peer);
            state[1] += count;
            state[9] += count;
            state[10] += PacketBytes(count);
            advanced = true;
          }
          const int32_t send_sequence = geometry[4] + send_offset / a_.quantum + 1;
          const int32_t send_lane = (send_sequence - 1) % a_.lanes;
          if (send_offset < geometry[1] && Test(1, peer, send_lane, send_sequence) &&
              ProbeInput(geometry[0] + send_offset, Groups(geometry[1] - send_offset), state)) {
            const int32_t count = Groups(geometry[1] - send_offset);
            auto *destination = reinterpret_cast<GM_ADDR>(
              hyper_parallel::multicore::shmem::cann::device::remote_ptr(Ring(a_.rank, send_lane), peer));
            const auto begin = GetSystemCycle();
            Pack(destination, geometry[0] + send_offset, count);
            state[13] += GetSystemCycle() - begin;
            auto *remote_header = reinterpret_cast<GM_ADDR>(
              hyper_parallel::multicore::shmem::cann::device::remote_ptr(Header(a_.rank, send_lane), peer));
            state[5] += WriteHeader(remote_header, send_sequence, count, geometry[0] + send_offset);
            communication::mte_rma::quiet();
            communication::signal_set(Signal(0, a_.rank, send_lane), send_sequence, peer);
            state[0] += count;
            state[6] += count;
            state[7] += PacketBytes(count);
            advanced = true;
          }
        }
        if (advanced) {
          state[12] = GetSystemCycle();
        } else if (GetSystemCycle() - state[12] > a_.watchdog) {
          state[4] = -1;
          Trap();
        }
        const bool incomplete = StripeOffset(state[0], geometry[4], stripe, stripes) < geometry[1] ||
          StripeOffset(state[1], geometry[5], stripe, stripes) < geometry[3];
        pending = pending || incomplete;
        if (!incomplete) {
          state[3] = GetSystemCycle();
        }
      }
    }
  }

 private:
  __aicore__ inline int64_t StripeOffset(int64_t processed, int64_t base, int32_t stripe, int32_t stripes) {
    const int64_t first = ((stripe - base % stripes + stripes) % stripes) * a_.quantum;
    return first + processed * stripes;
  }

  __aicore__ inline int32_t Groups(int64_t remaining) { return remaining < a_.quantum ? remaining : a_.quantum; }

  __aicore__ inline GM_ADDR Ring(int32_t peer, int32_t lane) {
    return reinterpret_cast<GM_ADDR>(a_.ring) +
      static_cast<int64_t>(peer * a_.lanes + lane) * a_.window_rows * a_.hidden * 2;
  }

  __aicore__ inline GM_ADDR Header(int32_t peer, int32_t lane) {
    return reinterpret_cast<GM_ADDR>(a_.headers) +
      static_cast<int64_t>(peer * a_.lanes + lane) * a_.header_words * sizeof(int64_t);
  }

  __aicore__ inline __gm__ int32_t *Signal(int32_t kind, int32_t peer, int32_t lane) {
    return reinterpret_cast<__gm__ int32_t *>(a_.signals) + ((kind * a_.ranks + peer) * a_.lanes + lane) * 16;
  }

  __aicore__ inline bool Test(int32_t kind, int32_t peer, int32_t lane, int32_t value) {
    return aclshmem_int32_test(Signal(kind, peer, lane), ACLSHMEM_CMP_EQ, value) != 0;
  }

  __aicore__ inline void Flush(__gm__ int32_t *address) {
    GlobalTensor<int32_t> line;
    line.SetGlobalBuffer(address, 16);
    __asm__ __volatile__("" ::: "memory");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(line);
    __asm__ __volatile__("" ::: "memory");
  }

  __aicore__ inline bool InputReady(int64_t group, int32_t count) {
    if (a_.input_ready == 0) {
      return true;
    }
    if (a_.experts <= 512 && static_cast<int64_t>(a_.producers) * a_.experts <= 16384) {
      return CachedInputReady(group, count);
    }
    auto *cells = reinterpret_cast<__gm__ int32_t *>(a_.input_ready);
    auto *expected = reinterpret_cast<__gm__ int32_t *>(a_.input_expected);
    const auto *slots = reinterpret_cast<__gm__ int32_t *>(a_.row_slots);
    for (int32_t producer = 0; producer < a_.producers; ++producer) {
      const bool cached = a_.experts <= 32768 / 64;
      auto snapshot = ready_cache_.Get<int32_t>();
      if (cached) {
        Load(snapshot, reinterpret_cast<GM_ADDR>(cells + static_cast<int64_t>(producer) * a_.experts * 16),
             a_.experts * 16);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
      }
      for (int32_t offset = 0; offset < count; ++offset) {
        for (int32_t member = 0; member < a_.top_k; ++member) {
          const int32_t row = source_members_[(group + offset) * a_.top_k + member];
          if (row < 0) {
            continue;
          }
          if (row >= a_.input_rows) {
            Trap();
          }
          const int32_t expert = slots[row * 2];
          if (expert < 0 || expert >= a_.experts) {
            Trap();
          }
          const int64_t index = static_cast<int64_t>(producer) * a_.experts + expert;
          auto *cell = cells + index * 16;
          if (cached) {
            auto *local = reinterpret_cast<__ubuf__ int32_t *>(snapshot.GetPhyAddr()) + expert * 16;
            const int32_t required = local[routed::kProgressMode] == 1
              ? routed::RequiredTaskProgress(local, row, row) : expected[index];
            if (local[0] < required) {
              return false;
            }
          } else {
            Flush(cell);
            const int32_t required = cell[routed::kProgressMode] == 1
              ? routed::RequiredTaskProgress(cell, row, row) : expected[index];
            if (cell[0] < required) {
              return false;
            }
          }
        }
      }
    }
    return true;
  }

  __aicore__ inline void ReadProducer(LocalTensor<int32_t> snapshot, int32_t producer) {
    auto *cells = reinterpret_cast<__gm__ int32_t *>(a_.input_ready);
    Load(snapshot, reinterpret_cast<GM_ADDR>(cells + static_cast<int64_t>(producer) * a_.experts * 16),
         a_.experts * 16);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
  }

  __aicore__ inline bool ProducerComplete(LocalTensor<int32_t> snapshot, int32_t producer) {
    auto *expected = reinterpret_cast<__gm__ int32_t *>(a_.input_expected);
    for (int32_t expert = 0; expert < a_.experts; ++expert) {
      auto *cell = reinterpret_cast<__ubuf__ int32_t *>(snapshot.GetPhyAddr()) + expert * 16;
      const int32_t total = cell[routed::kProgressMode] == 1
        ? cell[routed::kProgressTaskCount] : expected[producer * a_.experts + expert];
      if (cell[0] < total) {
        return false;
      }
    }
    return true;
  }

  __aicore__ inline bool CacheTaskGeometry() {
    if (task_geometry_status_ != 0) {
      return task_geometry_status_ == 1;
    }
    if (((static_cast<int64_t>(a_.producers) * a_.experts + 7) / 8) * 8 + a_.experts * 8 > 16384) {
      task_geometry_status_ = 2;
      return false;
    }
    auto required = required_cache_.Get<int32_t>();
    auto snapshot = ready_cache_.Get<int32_t>();
    auto *expected = reinterpret_cast<__gm__ int32_t *>(a_.input_expected);
    const int32_t base = (a_.producers * a_.experts + 7) / 8 * 8;
    Duplicate(required[base], 0, a_.experts * 8);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int32_t producer = 0; producer < a_.producers; ++producer) {
      ReadProducer(snapshot, producer);
      for (int32_t expert = 0; expert < a_.experts; ++expert) {
        if (expected[producer * a_.experts + expert] == 0) {
          continue;
        }
        auto *cell = reinterpret_cast<__ubuf__ int32_t *>(snapshot.GetPhyAddr()) + expert * 16;
        if (cell[routed::kProgressMode] != 1) {
          task_geometry_status_ = 2;
          return false;
        }
        const int32_t slot = base + expert * 8;
        if (required.GetValue(slot) == 0) {
          required.SetValue(slot, 1);
          required.SetValue(slot + 1, cell[routed::kProgressFirstRow]);
          required.SetValue(slot + 2, cell[routed::kProgressTileRows]);
          required.SetValue(slot + 3, cell[routed::kProgressColumns]);
          required.SetValue(slot + 4, cell[routed::kProgressFirstGlobal] - cell[routed::kProgressFirstTask]);
          required.SetValue(slot + 5, cell[routed::kProgressProducers]);
          required.SetValue(slot + 6, cell[routed::kProgressRows]);
        }
      }
    }
    task_geometry_status_ = 1;
    return true;
  }

  __aicore__ inline void BuildTaskRequirements(int64_t group, int32_t count) {
    auto required = required_cache_.Get<int32_t>();
    const auto *slots = reinterpret_cast<__gm__ int32_t *>(a_.row_slots);
    const int32_t base = (a_.producers * a_.experts + 7) / 8 * 8;
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Duplicate(required, 0, base);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    for (int32_t expert = 0; expert < a_.experts; ++expert) {
      required.SetValue(base + expert * 8 + 7, -1);
    }
    cached_empty_ = true;
    for (int32_t offset = 0; offset < count; ++offset) {
      for (int32_t member = 0; member < a_.top_k; ++member) {
        const int32_t row = source_members_[(group + offset) * a_.top_k + member];
        if (row < 0) {
          continue;
        }
        if (row >= a_.input_rows) {
          Trap();
        }
        const int32_t expert = slots[row * 2];
        if (expert < 0 || expert >= a_.experts) {
          Trap();
        }
        const int32_t slot = base + expert * 8;
        const int32_t origin = required.GetValue(slot + 1);
        const int32_t rows = required.GetValue(slot + 2);
        const int32_t columns = required.GetValue(slot + 3);
        const int32_t first = required.GetValue(slot + 4);
        const int32_t producers = required.GetValue(slot + 5);
        if (required.GetValue(slot) != 1 || rows < 1 || columns < 1 || first < 0 ||
            producers != a_.producers || row < origin || row - origin >= required.GetValue(slot + 6)) {
          Trap();
        }
        const int32_t tile = (row - origin) / rows;
        if (required.GetValue(slot + 7) == tile) {
          continue;
        }
        required.SetValue(slot + 7, tile);
        // Task ownership is (expert_first + local_task) % producers. Visit
        // each actual column task once, rather than scanning every producer.
        for (int32_t column = 0; column < columns; ++column) {
          const int64_t task = static_cast<int64_t>(tile) * columns + column;
          const int32_t producer = (first + task) % producers;
          const int32_t local_first = (producer - first % producers + producers) % producers;
          const int32_t needed = (task - local_first) / producers + 1;
          const int32_t index = producer * a_.experts + expert;
          if (needed > required.GetValue(index)) {
            required.SetValue(index, needed);
            cached_empty_ = false;
          }
        }
      }
    }
  }

  __aicore__ inline bool CachedInputReady(int64_t group, int32_t count) {
    auto required = required_cache_.Get<int32_t>();
    auto snapshot = ready_cache_.Get<int32_t>();
    auto *expected = reinterpret_cast<__gm__ int32_t *>(a_.input_expected);
    const auto *slots = reinterpret_cast<__gm__ int32_t *>(a_.row_slots);
    if ((cached_group_ != group || cached_count_ != count) && CacheTaskGeometry()) {
      BuildTaskRequirements(group, count);
      cached_group_ = group;
      cached_count_ = count;
    }
    if (cached_group_ != group || cached_count_ != count) {
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      Duplicate(required, 0, a_.producers * a_.experts);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      cached_empty_ = true;
      for (int32_t producer = 0; producer < a_.producers; ++producer) {
        ReadProducer(snapshot, producer);
        if (ProducerComplete(snapshot, producer)) {
          continue;
        }
        for (int32_t offset = 0; offset < count; ++offset) {
          for (int32_t member = 0; member < a_.top_k; ++member) {
            const int32_t row = source_members_[(group + offset) * a_.top_k + member];
            if (row < 0) {
              continue;
            }
            if (row >= a_.input_rows) {
              Trap();
            }
            const int32_t expert = slots[row * 2];
            if (expert < 0 || expert >= a_.experts) {
              Trap();
            }
            const int32_t index = producer * a_.experts + expert;
            auto *cell = reinterpret_cast<__ubuf__ int32_t *>(snapshot.GetPhyAddr()) + expert * 16;
            const int32_t needed = cell[routed::kProgressMode] == 1
              ? routed::RequiredTaskProgress(cell, row, row) : expected[index];
            if (needed > required.GetValue(index)) {
              required.SetValue(index, needed);
              cached_empty_ = false;
            }
          }
        }
      }
      // Task geometry is immutable after producer admission. Cache the exact
      // maximum for every producer/expert; only completed counts change.
      cached_group_ = group;
      cached_count_ = count;
    }
    if (cached_empty_) {
      return true;
    }
    for (int32_t producer = 0; producer < a_.producers; ++producer) {
      bool active = false;
      for (int32_t expert = 0; expert < a_.experts; ++expert) {
        active = active || required.GetValue(producer * a_.experts + expert) != 0;
      }
      if (!active) {
        continue;
      }
      ReadProducer(snapshot, producer);
      for (int32_t expert = 0; expert < a_.experts; ++expert) {
        const int32_t needed = required.GetValue(producer * a_.experts + expert);
        if (snapshot.GetValue(expert * 16) < needed) {
          return false;
        }
      }
    }
    return true;
  }

  __aicore__ inline bool ProbeInput(int64_t group, int32_t count, __gm__ int64_t *state) {
    const auto begin = GetSystemCycle();
    const bool ready = InputReady(group, count);
    state[15] += GetSystemCycle() - begin;
    return ready;
  }

  __aicore__ inline int64_t PacketBytes(int32_t groups) {
    return static_cast<int64_t>(groups) * a_.hidden * (wire_float_ ? 4 : 2) + 64 +
      ((a_.phase == 1 || a_.phase == 4) ? static_cast<int64_t>(groups) * scalar_stride_ * 4 : 0);
  }

  template <typename T>
  __aicore__ inline void Load(LocalTensor<T> destination, GM_ADDR address, uint32_t count) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    GlobalTensor<T> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address), count);
    DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0.0F)};
    DataCopyPad(destination, source, copy, padding);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  }

  template <typename T>
  __aicore__ inline void CopyOutput(GM_ADDR address, LocalTensor<T> source, uint32_t count) {
    GlobalTensor<T> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(address), count);
    DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPad(destination, source, copy);
  }

  template <typename T>
  __aicore__ inline void Store(GM_ADDR address, LocalTensor<T> source, uint32_t count) {
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    CopyOutput(address, source, count);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }

  __aicore__ inline void LoadFloat(LocalTensor<float> destination, GM_ADDR address, bool fp32, uint32_t count) {
    if (fp32) {
      Load(destination, address, count);
    } else {
      auto small = small_.Get<bfloat16_t>();
      Load(small, address, count);
      Cast(destination, small, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
    }
  }

  __aicore__ inline float Score(int32_t row) {
    if (row < 0 || row >= a_.score_rows) {
      Trap();
    }
    if (a_.phase == 1 && a_.score_bf16) {
      union Scalar {
        uint32_t bits;
        float value;
      } scalar;
      scalar.bits = static_cast<uint32_t>(reinterpret_cast<__gm__ uint16_t *>(a_.scores)[row]) << 16;
      return scalar.value;
    }
    return reinterpret_cast<__gm__ float *>(a_.scores)[row];
  }

  __aicore__ inline int64_t ScalarOffset(int32_t groups) {
    return (static_cast<int64_t>(groups) * a_.hidden * (wire_float_ ? 4 : 2) + 63) / 64 * 64;
  }

  template <int32_t Slot>
  __aicore__ inline void WaitPartial() {
    if (partial_pending_[Slot]) {
      WaitFlag<HardEvent::MTE3_V>(Slot == 0 ? EVENT_ID1 : EVENT_ID2);
      partial_pending_[Slot] = false;
    }
  }

  template <int32_t Slot>
  __aicore__ inline void StorePartial(GM_ADDR address, LocalTensor<float> values, uint32_t count) {
    GlobalTensor<float> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address), count);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopyExtParams copy{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(destination, values, copy);
    SetFlag<HardEvent::MTE3_V>(Slot == 0 ? EVENT_ID1 : EVENT_ID2);
    partial_pending_[Slot] = true;
  }

  __aicore__ inline int32_t NextMember(int64_t group, int32_t member) {
    while (member < a_.top_k && source_members_[group * a_.top_k + member] < 0) {
      ++member;
    }
    return member;
  }

  __aicore__ inline void LoadMember(int64_t group, int32_t member, int32_t column,
                                   uint32_t count, int32_t slot) {
    const int32_t row = source_members_[group * a_.top_k + member];
    if (row < 0 || row >= a_.input_rows) {
      Trap();
    }
    // The next input uses a distinct UB slot; only wait when reusing a slot
    // whose previous BF16 value may still be consumed by the vector pipeline.
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    GlobalTensor<bfloat16_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(a_.input) +
      static_cast<int64_t>(row) * a_.hidden + column, count);
    auto local = slot == 0 ? small_.Get<bfloat16_t>() : pack_small_.Get<bfloat16_t>();
    DataCopyExtParams copy{1, count * 2, 0, 0, 0};
    DataCopyPadExtParams<bfloat16_t> padding{false, 0, 0, static_cast<bfloat16_t>(0.0F)};
    DataCopyPad(local, source, copy, padding);
    SetFlag<HardEvent::MTE2_V>(slot == 0 ? EVENT_ID1 : EVENT_ID2);
  }

  __aicore__ inline void ReduceMembers(LocalTensor<float> accumulator, int64_t group,
                                      int32_t column, uint32_t count) {
    Duplicate(accumulator, 0.0F, count);
    PipeBarrier<PIPE_V>();
    int32_t member = NextMember(group, 0);
    if (member == a_.top_k) {
      return;
    }
    int32_t slot = 0;
    LoadMember(group, member, column, count, slot);
    while (member < a_.top_k) {
      const int32_t next = NextMember(group, member + 1);
      if (next < a_.top_k) {
        LoadMember(group, next, column, count, 1 - slot);
      }
      WaitFlag<HardEvent::MTE2_V>(slot == 0 ? EVENT_ID1 : EVENT_ID2);
      auto small = slot == 0 ? small_.Get<bfloat16_t>() : pack_small_.Get<bfloat16_t>();
      auto value = work_.Get<float>();
      Cast(value, small, RoundMode::CAST_NONE, count);
      PipeBarrier<PIPE_V>();
      if (a_.phase == 2) {
        Muls(value, value, Score(source_members_[group * a_.top_k + member]), count);
        PipeBarrier<PIPE_V>();
        if (a_.score_bf16) {
          Cast(small, value, RoundMode::CAST_RINT, count);
          PipeBarrier<PIPE_V>();
          Cast(value, small, RoundMode::CAST_NONE, count);
          PipeBarrier<PIPE_V>();
        }
      }
      Add(accumulator, accumulator, value, count);
      PipeBarrier<PIPE_V>();
      member = next;
      slot = 1 - slot;
    }
  }

  __aicore__ inline void Pack(GM_ADDR packet, int64_t first, int32_t groups) {
    int32_t tile = 0;
    for (int32_t offset = 0; offset < groups; ++offset) {
      const int64_t group = first + offset;
      for (int32_t column = 0; column < a_.hidden; column += kTile) {
        const uint32_t count = a_.hidden - column < kTile ? a_.hidden - column : kTile;
        if ((a_.phase == 1 || a_.phase == 3) && !a_.input_float) {
          const int32_t token = source_tokens_[group];
          if (token < 0 || token >= a_.input_rows) {
            Trap();
          }
          auto value = small_.Get<bfloat16_t>();
          Load(value, input_ + (static_cast<int64_t>(token) * a_.hidden + column) * 2, count);
          Store(packet + (static_cast<int64_t>(offset) * a_.hidden + column) * 2, value, count);
          continue;
        }
        const bool partial = a_.phase == 2 || a_.phase == 4;
        const bool second = partial && (tile++ % 2 != 0);
        if (partial) {
          if (second) {
            WaitPartial<1>();
          } else {
            WaitPartial<0>();
          }
        }
        auto accumulator = second ? pack_acc_.Get<float>() : acc_.Get<float>();
        if (a_.phase == 1 || a_.phase == 3) {
          const int32_t token = source_tokens_[group];
          if (token < 0 || token >= a_.input_rows) {
            Trap();
          }
          LoadFloat(accumulator, input_ + (static_cast<int64_t>(token) * a_.hidden + column) *
                    (a_.input_float ? 4 : 2), a_.input_float, count);
        } else {
          ReduceMembers(accumulator, group, column, count);
        }
        const int64_t index = static_cast<int64_t>(offset) * a_.hidden + column;
        if (partial) {
          if (second) {
            StorePartial<1>(packet + index * 4, accumulator, count);
          } else {
            StorePartial<0>(packet + index * 4, accumulator, count);
          }
        } else if (wire_float_) {
          Store(packet + index * 4, accumulator, count);
        } else {
          auto small = small_.Get<bfloat16_t>();
          Cast(small, accumulator, RoundMode::CAST_RINT, count);
          Store(packet + index * 2, small, count);
        }
      }
      if ((a_.phase == 1 || a_.phase == 4) && !cache_packet_scalars_) {
        for (int32_t begin = 0; begin < scalar_stride_; begin += 256) {
          const int32_t count = scalar_stride_ - begin < 256 ? scalar_stride_ - begin : 256;
          auto values = scalar_.Get<float>();
          for (int32_t member = 0; member < count; ++member) {
            const int32_t column = begin + member;
            const int32_t row = column < a_.top_k ? source_members_[group * a_.top_k + column] : -1;
            values.SetValue(member, row < 0 ? 0.0F : Score(row));
          }
          SetFlag<HardEvent::S_V>(EVENT_ID0);
          WaitFlag<HardEvent::S_V>(EVENT_ID0);
          Store(packet + ScalarOffset(groups) + (offset * scalar_stride_ + begin) * sizeof(float), values, count);
        }
      }
    }
    WaitPartial<0>();
    WaitPartial<1>();
    if (cache_packet_scalars_) {
      auto values = packet_scalars_.Get<float>();
      for (int32_t offset = 0; offset < groups; ++offset) {
        const int64_t group = first + offset;
        for (int32_t member = 0; member < scalar_stride_; ++member) {
          const int32_t row = member < a_.top_k ? source_members_[group * a_.top_k + member] : -1;
          values.SetValue(offset * scalar_stride_ + member, row < 0 ? 0.0F : Score(row));
        }
      }
      SetFlag<HardEvent::S_V>(EVENT_ID0);
      WaitFlag<HardEvent::S_V>(EVENT_ID0);
      Store(packet + ScalarOffset(groups), values, groups * scalar_stride_);
    }
    // Payload must be visible before the caller publishes this packet's header.
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }

  __aicore__ inline void BeginRows() {
    publish_count_ = 0;
    scalar_rows_count_ = 0;
    batch_rows_ = a_.output_ready != 0 && a_.input_ready == 0 &&
      static_cast<int64_t>(a_.ready_tiles) * a_.experts <= 8192;
    if (batch_rows_) {
      Duplicate(required_cache_.Get<int32_t>(), 0, a_.ready_tiles * a_.experts);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
  }

  __aicore__ inline void AddReady(int64_t cell, int32_t count) {
    auto increment = atomic_.Get<int32_t>();
    Duplicate(increment, 0, 8);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    increment.SetValue(0, count);
    GlobalTensor<int32_t> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(a_.output_ready) + cell * 16, 8);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    SetAtomicAdd<int32_t>();
    DataCopy(destination, increment, 8);
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
  }

  __aicore__ inline void FlushRows() {
    FlushScalarRows();
    if (!batch_rows_ || publish_count_ == 0) {
      return;
    }
    // Count only completed rows after their vectors and scores are globally visible.
    PipeBarrier<PIPE_ALL>();
    auto counts = required_cache_.Get<int32_t>();
    auto touched = ready_cache_.Get<int32_t>();
    for (int32_t index = 0; index < publish_count_; ++index) {
      const int32_t cell = touched.GetValue(index);
      AddReady(cell, counts.GetValue(cell));
    }
    publish_count_ = 0;
  }

  __aicore__ inline void PublishRow(int32_t row) {
    if (a_.output_ready == 0) {
      return;
    }
    const auto *slots = reinterpret_cast<__gm__ int32_t *>(a_.row_slots);
    const int32_t expert = slots[row * 2], tile = slots[row * 2 + 1];
    if (expert < 0 || expert >= a_.experts || tile < 0 || tile >= a_.ready_tiles) {
      Trap();
    }
    const int64_t cell = static_cast<int64_t>(tile) * a_.experts + expert;
    if (batch_rows_) {
      auto counts = required_cache_.Get<int32_t>();
      const int32_t previous = counts.GetValue(cell);
      if (previous == 0) {
        ready_cache_.Get<int32_t>().SetValue(publish_count_++, cell);
      }
      counts.SetValue(cell, previous + 1);
    } else {
      FlushScalarRows();
      PipeBarrier<PIPE_ALL>();
      AddReady(cell, 1);
    }
  }

  __aicore__ inline void FlushScalarRows() {
    if (scalar_rows_count_ == 0) {
      return;
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    GlobalTensor<float> destination;
    destination.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(a_.output_scores) + scalar_rows_first_,
                                scalar_rows_count_);
    DataCopyExtParams copy{1, static_cast<uint32_t>(scalar_rows_count_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(destination, scalar_rows_.Get<float>(), copy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    scalar_rows_count_ = 0;
  }

  __aicore__ inline void StoreScalar(int32_t row, float value) {
    if (row < 0 || row >= a_.output_score_rows) {
      Trap();
    }
    if (scalar_rows_count_ == 16 || (scalar_rows_count_ != 0 && row != scalar_rows_first_ + scalar_rows_count_)) {
      FlushScalarRows();
    }
    if (scalar_rows_count_ == 0) {
      scalar_rows_first_ = row;
    }
    scalar_rows_.Get<float>().SetValue(scalar_rows_count_++, value);
  }

  __aicore__ inline float PacketScalar(GM_ADDR packet, int32_t groups, int32_t offset, int32_t member) {
    auto values = packet_scalars_.Get<float>();
    if (cache_packet_scalars_) {
      return values.GetValue(offset * scalar_stride_ + member);
    }
    const int32_t begin = member / 256 * 256;
    if (scalar_group_ != offset || scalar_begin_ != begin) {
      const int32_t count = a_.top_k - begin < 256 ? a_.top_k - begin : 256;
      // Ring lanes are overwritten after credit return. Read the current
      // packet through MTE2 instead of reusing a scalar-cache line.
      Load(values, packet + ScalarOffset(groups) + (offset * scalar_stride_ + begin) * sizeof(float), count);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      scalar_group_ = offset;
      scalar_begin_ = begin;
    }
    return values.GetValue(member - begin);
  }

  __aicore__ inline void UnpackBackward(GM_ADDR packet, int64_t first, int32_t groups) {
    auto dots = packet_scalars_.Get<float>();
    for (int32_t offset = 0; offset < groups; ++offset) {
      const int64_t group = first + offset;
      Duplicate(dots, 0.0F, a_.top_k * 8);
      PipeBarrier<PIPE_V>();
      for (int32_t column = 0; column < a_.hidden; column += backward_tile_) {
        const uint32_t count = a_.hidden - column < backward_tile_ ? a_.hidden - column : backward_tile_;
        auto gradient = acc_.Get<float>();
        LoadFloat(gradient, packet + (static_cast<int64_t>(offset) * a_.hidden + column) *
                  (wire_float_ ? 4 : 2), wire_float_, count);
        for (int32_t member = 0; member < a_.top_k; ++member) {
          const int32_t row = target_members_[group * a_.top_k + member];
          if (row < 0) {
            continue;
          }
          if (row >= a_.output_rows) {
            Trap();
          }
          auto values = work_.Get<float>();
          LoadFloat(values, reinterpret_cast<GM_ADDR>(a_.saved) +
                    (static_cast<int64_t>(row) * a_.hidden + column) * 2, false, count);
          Mul(values, values, gradient, count);
          PipeBarrier<PIPE_V>();
          auto small = small_.Get<bfloat16_t>();
          if (a_.score_bf16 && !wire_float_) {
            Cast(small, values, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(values, small, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
          }
          auto reduced = scalar_.Get<float>();
          for (uint32_t begin = 0; begin < count; begin += kTile) {
            const uint32_t length = count - begin < kTile ? count - begin : kTile;
            ReduceSum(reduced, values[begin], other_.Get<float>(), length);
            PipeBarrier<PIPE_V>();
            // Preserve the original 4K reduction and left-to-right partial order.
            Add(dots[member * 8], dots[member * 8], reduced, 1);
            PipeBarrier<PIPE_V>();
          }
          const float probability = reinterpret_cast<__gm__ float *>(a_.target_scores)[row];
          Muls(values, gradient, probability, count);
          PipeBarrier<PIPE_V>();
          Cast(small, values, RoundMode::CAST_RINT, count);
          Store(output_ + (static_cast<int64_t>(row) * a_.hidden + column) * 2, small, count);
        }
      }
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      for (int32_t member = 0; member < a_.top_k; ++member) {
        const int32_t row = target_members_[group * a_.top_k + member];
        if (row >= 0) {
          StoreScalar(row, dots.GetValue(member * 8));
          PublishRow(row);
        }
      }
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline void Unpack(GM_ADDR packet, int64_t first, int32_t groups) {
    if (cache_packet_scalars_) {
      Load(packet_scalars_.Get<float>(), packet + ScalarOffset(groups), groups * scalar_stride_);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }
    BeginRows();
    // Reuse one received G tile across all experts. A bounded scalar table
    // retains each member's original left-to-right dot accumulation order.
    if (a_.phase == 3 && a_.top_k <= 256) {
      UnpackBackward(packet, first, groups);
      FlushRows();
      return;
    }
    scalar_group_ = -1;
    for (int32_t offset = 0; offset < groups; ++offset) {
      const int64_t group = first + offset;
      if (a_.phase == 1) {
        // Fanout reuses one received tile. Expert copies retain the original
        // BF16 bits and become visible before publishing their row counters.
        for (int32_t column = 0; column < a_.hidden; column += kTile) {
          const uint32_t count = a_.hidden - column < kTile ? a_.hidden - column : kTile;
          auto value = small_.Get<bfloat16_t>();
          Load(value, packet + (static_cast<int64_t>(offset) * a_.hidden + column) * 2, count);
          SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
          WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
          for (int32_t member = 0; member < a_.top_k; ++member) {
            const int32_t row = target_members_[group * a_.top_k + member];
            if (row < 0) {
              continue;
            }
            if (row >= a_.output_rows) {
              Trap();
            }
            CopyOutput(output_ + (static_cast<int64_t>(row) * a_.hidden + column) * 2, value, count);
          }
          // All destinations read the same immutable UB tile. Drain once
          // before loading the next tile or publishing any completed row.
          SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
          SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
          WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        }
        for (int32_t member = 0; member < a_.top_k; ++member) {
          const int32_t row = target_members_[group * a_.top_k + member];
          if (row >= 0) {
            StoreScalar(row, PacketScalar(packet, groups, offset, member));
            PublishRow(row);
          }
        }
        continue;
      }
      if (a_.phase == 2 || a_.phase == 4) {
        for (int32_t column = 0; column < a_.hidden; column += kTile) {
          const uint32_t count = a_.hidden - column < kTile ? a_.hidden - column : kTile;
          auto value = acc_.Get<float>();
          Load(value, packet + (static_cast<int64_t>(offset) * a_.hidden + column) * 4, count);
          Store(output_ + (group * a_.hidden + column) * 4, value, count);
        }
        if (a_.phase == 2) {
          continue;
        }
      }
      for (int32_t member = 0; member < a_.top_k; ++member) {
        const int32_t row = target_members_[group * a_.top_k + member];
        if (row < 0) {
          continue;
        }
        if (a_.phase == 2) {
          continue;
        }
        if (a_.phase == 4) {
          if (row >= a_.output_score_rows) {
            Trap();
          }
          StoreScalar(row, PacketScalar(packet, groups, offset, member));
          continue;
        }
        if (row >= a_.output_rows) {
          Trap();
        }
        const float probability = a_.phase == 1
          ? PacketScalar(packet, groups, offset, member)
          : reinterpret_cast<__gm__ float *>(a_.target_scores)[row];
        float dot = 0.0F;
        for (int32_t column = 0; column < a_.hidden; column += kTile) {
          const uint32_t count = a_.hidden - column < kTile ? a_.hidden - column : kTile;
          auto gradient = acc_.Get<float>();
          LoadFloat(gradient, packet + (static_cast<int64_t>(offset) * a_.hidden + column) *
                    (wire_float_ ? 4 : 2), wire_float_, count);
          auto small = small_.Get<bfloat16_t>();
          if (a_.phase == 3) {
            auto values = work_.Get<float>();
            LoadFloat(values, reinterpret_cast<GM_ADDR>(a_.saved) +
                      (static_cast<int64_t>(row) * a_.hidden + column) * 2, false, count);
            Mul(values, values, gradient, count);
            PipeBarrier<PIPE_V>();
            if (a_.score_bf16 && !wire_float_) {
              Cast(small, values, RoundMode::CAST_RINT, count);
              PipeBarrier<PIPE_V>();
              Cast(values, small, RoundMode::CAST_NONE, count);
              PipeBarrier<PIPE_V>();
            }
            auto reduced = scalar_.Get<float>();
            ReduceSum(reduced, values, other_.Get<float>(), count);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            dot += reduced.GetValue(0);
            Muls(gradient, gradient, probability, count);
            PipeBarrier<PIPE_V>();
          }
          Cast(small, gradient, RoundMode::CAST_RINT, count);
          Store(output_ + (static_cast<int64_t>(row) * a_.hidden + column) * 2, small, count);
        }
        StoreScalar(row, a_.phase == 1 ? probability : dot);
        PublishRow(row);
      }
    }
    PipeBarrier<PIPE_ALL>();
    FlushRows();
  }

  __aicore__ inline int64_t ScalarCount(const __gm__ int32_t *members, int64_t first, int32_t count) {
    int64_t total = 0;
    if (a_.phase == 1 || a_.phase == 4) {
      for (int32_t index = 0; index < count * a_.top_k; ++index) {
        total += members[first * a_.top_k + index] >= 0;
      }
    }
    return (static_cast<int64_t>(a_.top_k) << 32) | total;
  }

  __aicore__ inline int64_t WriteHeader(GM_ADDR address, int32_t sequence, int32_t count, int64_t first) {
    auto values = header_.Get<int64_t>();
    values.SetValue(0, kMagic);
    values.SetValue(1, a_.epoch);
    values.SetValue(2, a_.phase);
    values.SetValue(3, sequence);
    values.SetValue(4, count);
    values.SetValue(5, ScalarCount(source_members_, first, count));
    values.SetValue(6, wire_float_);
    values.SetValue(7, a_.score_bf16);
    SetFlag<HardEvent::S_V>(EVENT_ID0);
    WaitFlag<HardEvent::S_V>(EVENT_ID0);
    Store(address, values, 8);
    return values.GetValue(5) & 0xffffffffLL;
  }

  __aicore__ inline int64_t CheckHeader(GM_ADDR address, int32_t sequence, int32_t count, int64_t first) {
    auto values = header_.Get<int64_t>();
    Load(values, address, 8);
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    if (values.GetValue(0) != kMagic || values.GetValue(1) != a_.epoch || values.GetValue(2) != a_.phase ||
        values.GetValue(3) != sequence || values.GetValue(4) != count ||
        values.GetValue(5) != ScalarCount(target_members_, first, count) ||
        values.GetValue(6) != wire_float_ || values.GetValue(7) != a_.score_bf16) {
      Trap();
    }
    return values.GetValue(5) & 0xffffffffLL;
  }

  routed::GroupLaunch a_;
  GM_ADDR input_;
  GM_ADDR output_;
  __gm__ int32_t *source_tokens_, *source_members_, *target_members_;
  __gm__ int64_t *geometry_, *progress_;
  TPipe pipe_;
  TBuf<TPosition::VECCALC> acc_, pack_acc_, work_, other_, small_, pack_small_;
  TBuf<TPosition::VECCALC> header_, scalar_, packet_scalars_, atomic_, scalar_rows_;
  TBuf<TPosition::VECCALC> ready_cache_, required_cache_;
  int64_t cached_group_ = -1;
  int32_t cached_count_ = 0;
  int32_t task_geometry_status_ = 0;
  bool cached_empty_ = true;
  int32_t scalar_group_ = -1, scalar_begin_ = 0;
  int32_t scalar_stride_;
  uint32_t backward_tile_;
  bool partial_pending_[2] = {false, false};
  bool wire_float_;
  bool batch_rows_ = false;
  bool cache_packet_scalars_ = false;
  int32_t scalar_rows_first_ = 0, scalar_rows_count_ = 0;
  int32_t publish_count_ = 0;
};

template <int32_t Phase>
__global__ __aicore__ void GroupExchangeKernel(routed::GroupLaunch args) {
  args.phase = Phase;
  GroupWorker worker;
  worker.Init(args);
  worker.Run();
}
}  // namespace

extern "C" int RoutedExpertsGroupExchange(void *stream, const routed::GroupLaunch *args, int32_t workers) {
  if (args == nullptr || workers < 1 || workers > RoutedExpertsVectorCores() - 8 ||
      args->phase < 1 || args->phase > 4) {
    return -1;
  }
  switch (args->phase) {
    case 1: GroupExchangeKernel<1><<<workers, nullptr, stream>>>(*args); break;
    case 2: GroupExchangeKernel<2><<<workers, nullptr, stream>>>(*args); break;
    case 3: GroupExchangeKernel<3><<<workers, nullptr, stream>>>(*args); break;
    case 4: GroupExchangeKernel<4><<<workers, nullptr, stream>>>(*args); break;
  }
  return static_cast<int>(aclrtGetLastError(ACL_RT_THREAD_LEVEL));
}
