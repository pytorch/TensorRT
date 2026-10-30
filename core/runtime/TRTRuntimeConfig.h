/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "NvInfer.h"
#include "NvInferVersion.h"
#include "core/runtime/RuntimeSettings.h"

// nvinfer1::IRuntimeConfig, with ICudaEngine::createRuntimeConfig() and
// createExecutionContext(IRuntimeConfig*), is in TensorRT 10.11 and later and in
// TensorRT-RTX since 1.0. Decide from TensorRT's own version macros, so that all
// code built against the same TensorRT agrees, whether built by Bazel or not.
// TensorRT-RTX defines NV_TENSORRT_MAJOR as TRT_MAJOR_RTX, so test it first.
// Test with `#if TORCHTRT_HAS_IRUNTIME_CONFIG()`: where this header is missing,
// that fails to compile instead of reading as false, as `#ifdef` would.
#if defined(TRT_MAJOR_RTX)
#define TORCHTRT_HAS_IRUNTIME_CONFIG() 1
#elif NV_TENSORRT_MAJOR > 10 || (NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR >= 11)
#define TORCHTRT_HAS_IRUNTIME_CONFIG() 1
#else
#define TORCHTRT_HAS_IRUNTIME_CONFIG() 0
#endif

#if !TORCHTRT_HAS_IRUNTIME_CONFIG()
// Declared only where TensorRT lacks it, so that a later TensorRT may redefine
// it (as an alias, say) without conflicting. See TRTRuntimeConfig::config.
namespace nvinfer1 {
class IRuntimeConfig;
} // namespace nvinfer1
#endif

namespace torch_tensorrt {
namespace core {
namespace runtime {

struct TRTRuntimeConfig {
  TRTRuntimeConfig() = default;
  explicit TRTRuntimeConfig(RuntimeSettings settings) : settings_(std::move(settings)) {}

  // Canonical user-facing runtime settings for this engine.
  [[nodiscard]] RuntimeSettings const& settings() const noexcept {
    return settings_;
  }

  // Setter. The live `IRuntimeConfig` is invalidated and the next
  // `ensure_initialized` rebuilds.
  [[nodiscard]] bool settings(RuntimeSettings new_settings);

  // (Re)build the `IRuntimeConfig` from `settings_`. Idempotent if the previous
  // build was against identical settings.
  void ensure_initialized(nvinfer1::ICudaEngine* cuda_engine);

  // Force the next `ensure_initialized` to rebuild from scratch.
  void reset();

  [[nodiscard]] std::shared_ptr<nvinfer1::IExecutionContext> create_execution_context(
      nvinfer1::ICudaEngine* cuda_engine,
      nvinfer1::ExecutionContextAllocationStrategy allocation_strategy);

  // Returns true if TRT-RTX owns capture/replay for the current settings --
  // caller should then bypass its own `at::cuda::CUDAGraph` capture around
  // enqueueV3. Always false on non-RTX builds.
  [[nodiscard]] bool uses_internal_capture(bool cudagraphs_enabled) const noexcept;

  // Returns true iff the execution context can be safely included in an outer
  // monolithic capture. Non-RTX builds always return true. Not noexcept: the
  // RTX path asserts ``exec_ctx != nullptr`` via ``TORCHTRT_ASSERT`` which can
  // throw on assertion failure.
  [[nodiscard]] bool is_monolithic_capturable(
      bool has_dynamic_inputs,
      nvinfer1::IExecutionContext* exec_ctx,
      cudaStream_t stream) const;

  // Lazy-constructed live config. `nullptr` until first `ensure_initialized`,
  // and always when TORCHTRT_HAS_IRUNTIME_CONFIG() is 0. Declared in every
  // build, so that TRTEngine's layout does not depend on it (see TRTEngine.h).
  std::shared_ptr<nvinfer1::IRuntimeConfig> config;

 private:
  RuntimeSettings settings_;
};

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
