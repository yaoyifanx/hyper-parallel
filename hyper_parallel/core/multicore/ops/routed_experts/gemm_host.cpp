// Copyright 2026 Huawei Technologies Co., Ltd
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstring>
#include "tiling/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "transport.h"

namespace {

constexpr int32_t kTileM = 128;
constexpr int32_t kTileN = 256;
constexpr int32_t kTileK = 64;
constexpr int32_t kReductionSteps = 4;
constexpr uint64_t kL1Bytes = 2ULL * (kTileM + kTileN) * kTileK * kReductionSteps * sizeof(uint16_t);
constexpr uint64_t kL0CBytes = 1ULL * kTileM * kTileN * sizeof(float);

bool FitsBufferedTile(const platform_ascendc::PlatformAscendC &platform, int32_t m, int32_t n, int32_t k) {
  uint64_t l1_bytes = 0, l0a_bytes = 0, l0b_bytes = 0, l0c_bytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_bytes);
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_bytes);
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_bytes);
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_bytes);
  // L1/L0 usage depends on the fixed K tile, while the full reduction width
  // changes only its iteration count. Avoid selecting tiling by model shape.
  return (platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910B ||
          platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_93) &&
    m >= kTileM && n >= kTileN && k >= kTileK && k % kTileK == 0 &&
    l1_bytes >= kL1Bytes && l0a_bytes >= kTileM * kTileK * kReductionSteps * sizeof(uint16_t) &&
    l0b_bytes >= 2ULL * kTileN * kTileK * sizeof(uint16_t) && l0c_bytes >= kL0CBytes;
}

}  // namespace

extern "C" int32_t RoutedExpertsVectorCores() {
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  return platform == nullptr ? 0 : static_cast<int32_t>(platform->GetCoreNumAiv());
}

extern "C" uint64_t RoutedExpertsUnifiedBufferBytes() {
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  uint64_t bytes = 0;
  if (platform != nullptr) {
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, bytes);
  }
  return bytes;
}

extern "C" int32_t RoutedExpertsTransportWorkers(int32_t ranks) {
  const int32_t budget = RoutedExpertsVectorCores() - 8;
  if (ranks < 1 || ranks >= (1 << 30) || budget < 1) {
    return 0;
  }
  return 2 * ranks <= budget ? 2 * ranks : std::min(ranks, budget);
}

extern "C" uint64_t RoutedExpertsGemmConfigBytes() {
  return sizeof(AscendC::tiling::TCubeTiling);
}

extern "C" int RoutedExpertsGemmConfig(void *output, uint64_t capacity, int32_t m, int32_t n, int32_t k,
                                       bool transpose_b, uint64_t *written, uint64_t *workspace_bytes) {
  if (output == nullptr || written == nullptr || workspace_bytes == nullptr ||
      capacity < RoutedExpertsGemmConfigBytes() || m < 1 || n < 1 || k < 1) {
    return -1;
  }
  auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
  if (platform == nullptr) {
    return -1;
  }
  matmul_tiling::MatmulApiTiling builder(*platform);
  builder.SetMatmulConfigParams(1);
  builder.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_BFLOAT16, false);
  builder.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_BFLOAT16, transpose_b);
  builder.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                   matmul_tiling::DataType::DT_BFLOAT16);
  builder.SetShape(m, n, k);
  builder.SetOrgShape(m, n, k);
  builder.SetBias(false);
  builder.SetBufferSpace(-1, -1, -1);
  const bool fixed_tiling = FitsBufferedTile(*platform, m, n, k);
  if (fixed_tiling) {
    if (builder.SetFixSplit(kTileM, kTileN, kTileK) != 0) {
      return -2;
    }
    builder.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM);
  }
  AscendC::tiling::TCubeTiling tiling;
  if (builder.GetTiling(tiling) != 0) {
    return -2;
  }
  if (fixed_tiling) {
    // Keep both operands double-buffered within the platform-reported usable L1.
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = kReductionSteps;
    tiling.stepKb = kReductionSteps;
    tiling.depthA1 = 2 * kReductionSteps;
    tiling.depthB1 = 2 * kReductionSteps;
    tiling.dbL0C = 1;
    tiling.shareL1Size = kL1Bytes;
    tiling.shareL0CSize = kL0CBytes;
  }
  std::memcpy(output, &tiling, sizeof(tiling));
  *written = sizeof(tiling);
  // The cube-only MatmulImpl does not use KFC service workspace.
  *workspace_bytes = 0;
  return 0;
}
