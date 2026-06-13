#include "core/runtime/TRTRuntimeConfig.h"

#include <sstream>
#include <stdexcept>
#include <utility>

#include "core/runtime/RuntimeSettings.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

bool TRTRuntimeConfig::settings(RuntimeSettings new_settings) {
  if (new_settings == settings_) {
    return false;
  }
  settings_ = std::move(new_settings);
  // Invalidate the live IRuntimeConfig so the next `ensure_initialized` rebuilds
  // with the new strategy values + cache attachment.
  reset();
  return true;
}

void TRTRuntimeConfig::ensure_initialized(TORCHTRT_UNUSED nvinfer1::ICudaEngine* cuda_engine) {
#ifdef TRT_HAS_IRUNTIME_CONFIG
  if (!config) {
    TORCHTRT_CHECK(cuda_engine != nullptr, "Cannot initialize TRTRuntimeConfig without a live ICudaEngine");
    config = make_trt(cuda_engine->createRuntimeConfig());
    TORCHTRT_CHECK(config.get() != nullptr, "Unable to create TensorRT IRuntimeConfig");
  }

#ifdef TRT_MAJOR_RTX
  // Runtime cache: ONLY attach when the caller provided an external
  // RuntimeCacheHandle. The Python TRTEngine side creates an implicit
  // handle from a path string and passes it in via the handle; without
  // an explicit user opt-in we leave the IRuntimeConfig cache-less.
  auto& rt_cache = settings_.runtime_cache;
  if (rt_cache) {
    // ``cache`` is owned by ``rt_cache->trt_handle_`` (kept alive through the
    // settings); the local ``shared_ptr`` here is just an attach handle and
    // does not need to outlive the call.
    auto cache = rt_cache->ensure_materialized(config.get());
    TORCHTRT_CHECK(cache.get() != nullptr, "Failed to create IRuntimeCache for shared RuntimeCacheHandle");
    if (config->setRuntimeCache(*cache)) {
      LOG_DEBUG("Attached external IRuntimeCache to IRuntimeConfig.");
    } else {
      LOG_WARNING("Failed to attach IRuntimeCache to IRuntimeConfig; cache will be unused.");
    }
  } else {
    LOG_DEBUG("Runtime cache disabled (no RuntimeCacheHandle provided).");
  }

  // Enum values mirror the nvinfer1 enums byte-for-byte (validated at the
  // boundary via ``from_underlying``); a plain ``static_cast`` is enough,
  // routed through the wrapper's implicit ``operator Value()`` unwrap.
  config->setDynamicShapesKernelSpecializationStrategy(static_cast<nvinfer1::DynamicShapesKernelSpecializationStrategy>(
      settings_.dynamic_shapes_kernel_specialization_strategy));
  LOG_DEBUG(
      "Dynamic shapes kernel specialization strategy set to "
      << settings_.dynamic_shapes_kernel_specialization_strategy.to_string());

  if (!config->setCudaGraphStrategy(static_cast<nvinfer1::CudaGraphStrategy>(settings_.cuda_graph_strategy))) {
    LOG_WARNING("Failed to set CUDA graph strategy; continuing with default.");
  }
#endif
#endif // TRT_HAS_IRUNTIME_CONFIG
}

void TRTRuntimeConfig::reset() {
#ifdef TRT_HAS_IRUNTIME_CONFIG
  config.reset();
#endif
}

std::shared_ptr<nvinfer1::IExecutionContext> TRTRuntimeConfig::create_execution_context(
    nvinfer1::ICudaEngine* cuda_engine,
    nvinfer1::ExecutionContextAllocationStrategy allocation_strategy) {
  ensure_initialized(cuda_engine);
#ifdef TRT_HAS_IRUNTIME_CONFIG
  config->setExecutionContextAllocationStrategy(allocation_strategy);
  return make_trt(cuda_engine->createExecutionContext(config.get()));
#else
  // Pre-10.11 TRT (e.g. Jetpack): use the legacy strategy overload directly.
  return make_trt(cuda_engine->createExecutionContext(allocation_strategy));
#endif
}

bool TRTRuntimeConfig::uses_internal_capture(TORCHTRT_UNUSED bool cudagraphs_enabled) const noexcept {
#ifdef TRT_MAJOR_RTX
  // On TRT-RTX the internal runtime handles capture/replay whenever a non-disabled
  // strategy is set, or when subgraph cudagraphs are enabled globally. In both
  // cases the caller should skip its manual at::cuda::CUDAGraph wrapper.
  return settings_.cuda_graph_strategy != CudaGraphStrategy::kDISABLED || cudagraphs_enabled;
#else
  return false;
#endif
}

bool TRTRuntimeConfig::is_monolithic_capturable(
    TORCHTRT_UNUSED bool has_dynamic_inputs,
    TORCHTRT_UNUSED nvinfer1::IExecutionContext* exec_ctx,
    TORCHTRT_UNUSED cudaStream_t stream) const {
#ifdef TRT_MAJOR_RTX
  TORCHTRT_ASSERT(exec_ctx != nullptr, "is_monolithic_capturable requires a live IExecutionContext");
  if (!exec_ctx->isStreamCapturable(stream)) {
    return false;
  }
  // "lazy" kernel specialization only swaps specialized kernels mid-run when an
  // input has a dynamic dimension; for static-shape engines the kernels are fixed
  // at setup and the captured graph stays valid. Mirrors the Python check.
  return !(
      settings_.dynamic_shapes_kernel_specialization_strategy == DynamicShapesKernelSpecializationStrategy::kLAZY &&
      has_dynamic_inputs);
#else
  return true;
#endif
}

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg) {
  os << "TRTRuntimeConfig{settings=" << cfg.settings().to_str();
#ifdef TRT_HAS_IRUNTIME_CONFIG
  os << ", config=" << (cfg.config ? "live" : "null");
#endif
  os << "}";
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
