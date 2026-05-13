#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "NvInfer.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// TensorRT-RTX-only configuration for how shape-specialized kernels are compiled.
enum class DynamicShapesKernelStrategy : int32_t {
  kLazy = 0,
  kEager = 1,
  kNone = 2,
};

// TensorRT-RTX-only configuration for how CUDA graph capture/replay is handled.
enum class CudaGraphStrategyOption : int32_t {
  kDisabled = 0,
  kWholeGraphCapture = 1,
};

// Encapsulates the IRuntimeConfig and TRT-RTX runtime state for a TRTEngine.
// IRuntimeConfig and runtime-cache `#ifdef`s are confined to this TU; serialization-
// index plumbing keeps its own RTX gates elsewhere.
struct TRTRuntimeConfig {
  // Settings - typically populated from engine deserialization before `ensure_initialized`.
  std::string runtime_cache_path = "";
  DynamicShapesKernelStrategy dynamic_shapes_kernel_strategy = DynamicShapesKernelStrategy::kLazy;
  CudaGraphStrategyOption cuda_graph_strategy = CudaGraphStrategyOption::kDisabled;

  // One-shot: set to true once an outer stream capture has been detected and the
  // engine-internal CUDA graph strategy has been disabled for the remainder of the
  // owning engine's lifetime.
  bool rtx_native_cudagraphs_disabled = false;

  // Live resources. The IRuntimeConfig is lazy-constructed on first `ensure_initialized`
  // and is unavailable on TensorRT versions older than 10.11 (e.g. Jetpack).
#ifdef TRT_HAS_IRUNTIME_CONFIG
  std::shared_ptr<nvinfer1::IRuntimeConfig> config;
#endif
#ifdef TRT_MAJOR_RTX
  std::shared_ptr<nvinfer1::IRuntimeCache> runtime_cache;
#endif

  // Lazily construct the IRuntimeConfig and apply RTX-specific settings. Idempotent.
  // No-op on builds without IRuntimeConfig (e.g. Jetpack).
  void ensure_initialized(nvinfer1::ICudaEngine* cuda_engine);

  // Lazy-initialize the IRuntimeConfig if needed and create an IExecutionContext that
  // honors `allocation_strategy`. Selects the right `createExecutionContext` overload
  // (IRuntimeConfig* vs ExecutionContextAllocationStrategy) so callers stay free of
  // any TRT_HAS_IRUNTIME_CONFIG branching.
  [[nodiscard]] std::shared_ptr<nvinfer1::IExecutionContext> create_execution_context(
      nvinfer1::ICudaEngine* cuda_engine,
      nvinfer1::ExecutionContextAllocationStrategy allocation_strategy);

  // Returns true if the TensorRT-RTX runtime owns capture/replay for this engine so the
  // caller should bypass its own at::cuda::CUDAGraph capture around enqueueV3. Always
  // false on non-RTX builds.
  [[nodiscard]] bool uses_internal_capture(bool cudagraphs_enabled) const;

  // One-shot: disable engine-internal CUDA graph capture. Invoked when an outer stream
  // capture is detected around execute_engine, so the outer capture can contain the
  // kernel launches directly. Saves the runtime cache before recreating the context so
  // compiled kernels from the present run are preserved for future reloads.
  void disable_rtx_native_cudagraphs(const std::string& engine_name) noexcept;

  // Whether the execution context is safe to include in an outer monolithic capture.
  // Non-RTX builds always return true.
  [[nodiscard]] bool is_monolithic_capturable(nvinfer1::IExecutionContext* exec_ctx, cudaStream_t stream) const;

  // Save the runtime cache to disk. Signature is `noexcept` so this is safe from a
  // destructor. The underlying file I/O is performed by free functions declared below
  // (non-noexcept, exception-leaky for easier testing); this member wraps them and
  // swallows any exceptions.
  void save_runtime_cache() noexcept;

  // Returns a human-readable summary of the runtime config.
  [[nodiscard]] std::string to_str() const;
};

// Construct a TRTRuntimeConfig from a flattened serialization vector. Reads the
// RTX-only indices only on RTX builds; standard TRT builds return a default-initialized
// struct.
[[nodiscard]] TRTRuntimeConfig make_runtime_config_from_serialized(const std::vector<std::string>& info);

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
