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
#include "core/runtime/RuntimeSettings.h"

namespace nvinfer1 {
class IRuntimeConfig; // TensorRT 10.11 and later; see TRTRuntimeConfig::config
} // namespace nvinfer1

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
  // and always without TRT_HAS_IRUNTIME_CONFIG. Declared in every build, so
  // that TRTEngine's layout does not depend on that macro (see TRTEngine.h).
  std::shared_ptr<nvinfer1::IRuntimeConfig> config;

 private:
  RuntimeSettings settings_;
};

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
