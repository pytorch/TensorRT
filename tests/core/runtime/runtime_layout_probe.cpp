/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstddef>
#include <cstdio>

#include "core/runtime/Platform.h"
#include "core/runtime/RTDevice.h"
#include "core/runtime/TRTEngine.h"

// Built as separate executables by tests/py/core/test_runtime_layout.py. Never
// link differently configured definitions of these classes into one process.
// No engine is constructed: this checks the native ABI without a GPU or a plan.
int main() {
  using torch_tensorrt::core::runtime::TRTEngine;
  using torch_tensorrt::core::runtime::TRTRuntimeConfig;

  std::printf("TRTEngine.size=%zu\n", sizeof(TRTEngine));
  std::printf("TRTEngine.alignment=%zu\n", alignof(TRTEngine));
  std::printf("TRTRuntimeConfig.size=%zu\n", sizeof(TRTRuntimeConfig));
  std::printf("TRTRuntimeConfig.alignment=%zu\n", alignof(TRTRuntimeConfig));

  // These classes are not standard-layout. GCC and Clang support offsetof on
  // them; the driver suppresses only the corresponding portability warning.
  // Size alone misses the NCCL bool fitting into padding while shifting the
  // following bool. Check that offset and later members too.
  std::printf("TRTEngine.profile_execution=%zu\n", offsetof(TRTEngine, profile_execution));
  std::printf("TRTEngine.device_profile_path=%zu\n", offsetof(TRTEngine, device_profile_path));
  std::printf("TRTEngine.runtime_cfg=%zu\n", offsetof(TRTEngine, runtime_cfg));
}
