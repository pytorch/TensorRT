#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

#include "NvInfer.h"
#include "core/runtime/RuntimeSettings.h"

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

#ifdef TRT_HAS_IRUNTIME_CONFIG
  // Lazy-constructed live config. `nullptr` until first `ensure_initialized`.
  std::shared_ptr<nvinfer1::IRuntimeConfig> config;
#endif

 private:
  RuntimeSettings settings_;
};

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
