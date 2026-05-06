#include "core/runtime/TRTRuntimeConfig.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// File-local helpers. Kept out of the header because they are only used by this
// translation unit -- TRTEngine now consumes a TRTRuntimeConfig directly and does not
// need the enum conversion helpers.
namespace {

[[nodiscard]] std::string to_string(DynamicShapesKernelStrategy s) {
  switch (s) {
    case DynamicShapesKernelStrategy::kLazy:
      return "lazy";
    case DynamicShapesKernelStrategy::kEager:
      return "eager";
    case DynamicShapesKernelStrategy::kNone:
      return "none";
  }
  TORCHTRT_CHECK(
      false,
      "Unexpected DynamicShapesKernelStrategy value: "
          << static_cast<std::underlying_type_t<DynamicShapesKernelStrategy>>(s));
}

[[nodiscard]] std::string to_string(CudaGraphStrategyOption s) {
  switch (s) {
    case CudaGraphStrategyOption::kDisabled:
      return "disabled";
    case CudaGraphStrategyOption::kWholeGraphCapture:
      return "whole_graph_capture";
  }
  TORCHTRT_CHECK(
      false,
      "Unexpected CudaGraphStrategyOption value: " << static_cast<std::underlying_type_t<CudaGraphStrategyOption>>(s));
}

[[nodiscard]] DynamicShapesKernelStrategy to_dynamic_shapes_kernel_strategy(
    std::underlying_type_t<DynamicShapesKernelStrategy> v) {
  TORCHTRT_CHECK(
      v >= 0 && v <= 2,
      "Invalid dynamic shapes kernel strategy value: " << v << ". Expected 0 (lazy), 1 (eager), or 2 (none).");
  return static_cast<DynamicShapesKernelStrategy>(v);
}

[[nodiscard]] CudaGraphStrategyOption to_cuda_graph_strategy_option(std::underlying_type_t<CudaGraphStrategyOption> v) {
  TORCHTRT_CHECK(
      v >= 0 && v <= 1,
      "Invalid CUDA graph strategy value: " << v << ". Expected 0 (disabled) or 1 (whole_graph_capture).");
  return static_cast<CudaGraphStrategyOption>(v);
}

#ifdef TRT_MAJOR_RTX
// Raw cache I/O helpers. Exception-propagating; the caller wraps in try/catch at the
// TRTRuntimeConfig member level. Kept file-local because the IRuntimeCache type is
// itself TensorRT-RTX-only and tests reach this path through the member wrappers.
void load_runtime_cache(const std::string& path, nvinfer1::IRuntimeCache* cache) {
  TORCHTRT_CHECK(cache != nullptr, "load_runtime_cache requires a non-null IRuntimeCache");
  if (!std::filesystem::exists(path)) {
    LOG_DEBUG("No existing runtime cache at " << path);
    return;
  }
  std::ifstream f(path, std::ios::binary);
  std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.empty()) {
    return;
  }
  TORCHTRT_CHECK(cache->deserialize(buf.data(), buf.size()), "IRuntimeCache::deserialize returned false for " << path);
  LOG_INFO("Loaded runtime cache from " << path << " (" << buf.size() << " bytes)");
}

void save_runtime_cache_impl(const std::string& path, nvinfer1::IRuntimeCache* cache) {
  TORCHTRT_CHECK(cache != nullptr, "save_runtime_cache requires a non-null IRuntimeCache");
  auto host_mem = make_trt(cache->serialize());
  if (!host_mem || host_mem->size() == 0) {
    return;
  }
  std::filesystem::path fs_path(path);
  if (fs_path.has_parent_path()) {
    std::filesystem::create_directories(fs_path.parent_path());
  }
  std::filesystem::path tmp_path = fs_path;
  tmp_path += ".tmp";
  {
    std::ofstream out(tmp_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(host_mem->data()), host_mem->size());
  }
  std::filesystem::rename(tmp_path, fs_path);
  LOG_INFO("Saved runtime cache to " << path << " (" << host_mem->size() << " bytes)");
}
#endif // TRT_MAJOR_RTX

} // namespace

void TRTRuntimeConfig::ensure_initialized(nvinfer1::ICudaEngine* cuda_engine) {
  if (config) {
    return;
  }
  TORCHTRT_CHECK(cuda_engine != nullptr, "Cannot initialize TRTRuntimeConfig without a live ICudaEngine");
  config = make_trt(cuda_engine->createRuntimeConfig());
  TORCHTRT_CHECK(config.get() != nullptr, "Unable to create TensorRT IRuntimeConfig");

#ifdef TRT_MAJOR_RTX
  // Runtime cache -- TRT-RTX only.
  if (!runtime_cache_path.empty()) {
    runtime_cache = make_trt(config->createRuntimeCache());
    if (runtime_cache.get() == nullptr) {
      LOG_WARNING("Failed to create TensorRT IRuntimeCache; runtime cache will be skipped.");
    } else {
      try {
        load_runtime_cache(runtime_cache_path, runtime_cache.get());
      } catch (const std::exception& e) {
        LOG_WARNING("Failed to load runtime cache from " << runtime_cache_path << ": " << e.what());
      }
      if (config->setRuntimeCache(*runtime_cache)) {
        LOG_DEBUG("TensorRT-RTX runtime cache configured at " << runtime_cache_path);
      } else {
        LOG_WARNING("Failed to attach runtime cache to IRuntimeConfig; cache will be unused.");
        runtime_cache.reset();
      }
    }
  } else {
    LOG_DEBUG("Runtime cache disabled (no path configured).");
  }

  // Dynamic shapes kernel specialization strategy -- TRT-RTX only.
  config->setDynamicShapesKernelSpecializationStrategy(
      static_cast<nvinfer1::DynamicShapesKernelSpecializationStrategy>(dynamic_shapes_kernel_strategy));
  LOG_DEBUG("Dynamic shapes kernel specialization strategy set to " << to_string(dynamic_shapes_kernel_strategy));

  // CUDA graph strategy -- TRT-RTX only.
  if (!config->setCudaGraphStrategy(
          cuda_graph_strategy == CudaGraphStrategyOption::kWholeGraphCapture
              ? nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE
              : nvinfer1::CudaGraphStrategy::kDISABLED)) {
    LOG_WARNING("Failed to set CUDA graph strategy; continuing with default.");
  }
#endif
}

void TRTRuntimeConfig::set_execution_context_allocation_strategy(
    nvinfer1::ExecutionContextAllocationStrategy strategy) const {
  TORCHTRT_ASSERT(config, "TRTRuntimeConfig::config must be initialized before setting allocation strategy");
  config->setExecutionContextAllocationStrategy(strategy);
}

bool TRTRuntimeConfig::uses_internal_capture(TORCHTRT_UNUSED bool cudagraphs_enabled) const {
#ifdef TRT_MAJOR_RTX
  // On TRT-RTX the internal runtime handles capture/replay whenever a non-disabled
  // strategy is set, or when subgraph cudagraphs are enabled globally. In both cases the
  // caller should skip its manual at::cuda::CUDAGraph wrapper because TRT-RTX's internal
  // capture would collide with it.
  return cuda_graph_strategy != CudaGraphStrategyOption::kDisabled || cudagraphs_enabled;
#else
  return false;
#endif
}

void TRTRuntimeConfig::disable_rtx_native_cudagraphs(TORCHTRT_UNUSED const std::string& engine_name) noexcept {
#ifdef TRT_MAJOR_RTX
  if (rtx_native_cudagraphs_disabled || cuda_graph_strategy == CudaGraphStrategyOption::kDisabled) {
    return;
  }
  LOG_WARNING(
      "Outer CUDA stream capture detected; disabling TensorRT-RTX native CUDA graph strategy on engine "
      << engine_name << " for the remainder of its lifetime.");
  // Persist any kernels the engine-internal capture has compiled so far; the outer
  // capture will run without them otherwise, and we want future reloads to reuse them.
  save_runtime_cache();
  cuda_graph_strategy = CudaGraphStrategyOption::kDisabled;
  if (config && !config->setCudaGraphStrategy(nvinfer1::CudaGraphStrategy::kDISABLED)) {
    LOG_WARNING("Failed to update CUDA graph strategy on IRuntimeConfig after disable.");
  }
  rtx_native_cudagraphs_disabled = true;
#endif
}

bool TRTRuntimeConfig::is_monolithic_capturable(
    TORCHTRT_UNUSED nvinfer1::IExecutionContext* exec_ctx,
    TORCHTRT_UNUSED cudaStream_t stream) const {
#ifdef TRT_MAJOR_RTX
  TORCHTRT_ASSERT(exec_ctx != nullptr, "is_monolithic_capturable requires a live IExecutionContext");
  // "lazy" kernel specialization swaps specialized kernels in mid-run, which invalidates
  // captured graphs. Other strategies (eager/none) are safe when the context reports the
  // stream capturable.
  return exec_ctx->isStreamCapturable(stream) && dynamic_shapes_kernel_strategy != DynamicShapesKernelStrategy::kLazy;
#else
  return true;
#endif
}

void TRTRuntimeConfig::save_runtime_cache() noexcept {
#ifdef TRT_MAJOR_RTX
  if (!runtime_cache || runtime_cache_path.empty()) {
    return;
  }
  try {
    save_runtime_cache_impl(runtime_cache_path, runtime_cache.get());
  } catch (const std::exception& e) {
    LOG_WARNING("Failed to save runtime cache to " << runtime_cache_path << ": " << e.what());
  } catch (...) {
    LOG_WARNING("Failed to save runtime cache (unknown exception).");
  }
#endif
}

std::string TRTRuntimeConfig::to_str() const {
  std::ostringstream os;
  os << "Runtime Cache Path: " << (runtime_cache_path.empty() ? "<disabled>" : runtime_cache_path) << std::endl;
  os << "Dynamic Shapes Kernel Strategy: " << to_string(dynamic_shapes_kernel_strategy) << std::endl;
  os << "CUDA Graph Strategy: " << to_string(cuda_graph_strategy) << std::endl;
  return os.str();
}

TRTRuntimeConfig make_runtime_config_from_serialized(TORCHTRT_UNUSED const std::vector<std::string>& info) {
  TRTRuntimeConfig cfg;
#ifdef TRT_MAJOR_RTX
  cfg.runtime_cache_path = info[RUNTIME_CACHE_PATH_IDX];
  cfg.dynamic_shapes_kernel_strategy =
      to_dynamic_shapes_kernel_strategy(std::stoi(info[DYNAMIC_SHAPES_KERNEL_STRATEGY_IDX]));
  cfg.cuda_graph_strategy = to_cuda_graph_strategy_option(std::stoi(info[CUDA_GRAPH_STRATEGY_IDX]));
#endif
  return cfg;
}

std::ostream& operator<<(std::ostream& os, const TRTRuntimeConfig& cfg) {
  os << "Runtime cfg {" << std::endl;
  os << cfg.to_str();
  os << "}" << std::endl;
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
