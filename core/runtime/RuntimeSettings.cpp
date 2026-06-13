#include "core/runtime/RuntimeSettings.h"

#include <array>
#include <cstring>
#include <sstream>
#include <tuple>
#include <type_traits>

// ``at::empty`` (factory function) lives in ``ATen/Functions.h``. Some bundled
// torch builds (e.g. torch_l4t on Jetpack) only ship the minimal
// ``ATen/core/Tensor.h`` transitively, so include the full ATen surface here.
#include "ATen/ATen.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

namespace {

// Reverse-lookup tables. Indices match the enum integer values (which mirror
// the nvinfer1 enums). Out-of-range -> "<unknown>".
constexpr std::array<std::string_view, 3> kDsStrategyNames = {"lazy", "eager", "none"};
constexpr std::array<std::string_view, 2> kCgStrategyNames = {"disabled", "whole_graph_capture"};

} // namespace

// ---- DynamicShapesKernelSpecializationStrategy -----------------------------

std::string_view DynamicShapesKernelSpecializationStrategy::to_string() const noexcept {
  // Negative underlying values wrap to a huge ``size_t``, so a single bounds
  // check from the top covers both ends without needing ``std::clamp``.
  auto const i = static_cast<size_t>(v_);
  return i < std::size(kDsStrategyNames) ? kDsStrategyNames[i] : std::string_view{"<unknown>"};
}

DynamicShapesKernelSpecializationStrategy DynamicShapesKernelSpecializationStrategy::from_underlying(int64_t v) {
  TORCHTRT_CHECK(
      v >= 0 && static_cast<size_t>(v) < std::size(kDsStrategyNames),
      "Invalid dynamic_shapes_kernel_specialization_strategy int: " << v
                                                                    << " (expected 0..2 mapping to lazy|eager|none)");
  return DynamicShapesKernelSpecializationStrategy(static_cast<Value>(v));
}

DynamicShapesKernelSpecializationStrategy DynamicShapesKernelSpecializationStrategy::from_string(
    std::string_view name) {
  for (size_t i = 0; i < std::size(kDsStrategyNames); ++i) {
    if (kDsStrategyNames[i] == name) {
      return DynamicShapesKernelSpecializationStrategy(static_cast<Value>(i));
    }
  }
  TORCHTRT_CHECK(
      false, "Invalid dynamic_shapes_kernel_specialization_strategy name: " << name << " (expected lazy|eager|none)");
}

// ---- CudaGraphStrategy -----------------------------------------------------

std::string_view CudaGraphStrategy::to_string() const noexcept {
  auto const i = static_cast<size_t>(v_);
  return i < std::size(kCgStrategyNames) ? kCgStrategyNames[i] : std::string_view{"<unknown>"};
}

CudaGraphStrategy CudaGraphStrategy::from_underlying(int64_t v) {
  TORCHTRT_CHECK(
      v >= 0 && static_cast<size_t>(v) < std::size(kCgStrategyNames),
      "Invalid cuda_graph_strategy int: " << v << " (expected 0..1 mapping to disabled|whole_graph_capture)");
  return CudaGraphStrategy(static_cast<Value>(v));
}

CudaGraphStrategy CudaGraphStrategy::from_string(std::string_view name) {
  for (size_t i = 0; i < std::size(kCgStrategyNames); ++i) {
    if (kCgStrategyNames[i] == name) {
      return CudaGraphStrategy(static_cast<Value>(i));
    }
  }
  TORCHTRT_CHECK(false, "Invalid cuda_graph_strategy name: " << name << " (expected disabled|whole_graph_capture)");
}

// ---- RuntimeCacheHandle methods ---------------------------------------------
//
// The ``#ifdef TRT_MAJOR_RTX`` is intentionally confined to this translation
// unit: the public header advertises a uniform interface (always-callable
// methods that simply degrade to no-ops on non-RTX builds), and the JIT-binding
// registration file (``register_jit_hooks.cpp``) calls these as plain member
// references with zero conditional compilation.

at::Tensor RuntimeCacheHandle::serialize() const {
  auto const opts = at::TensorOptions().dtype(at::kByte);
  auto const empty = [&]() { return at::empty({0}, opts); };
#ifdef TRT_MAJOR_RTX
  std::lock_guard<std::mutex> lock(state_mu_);
  if (!trt_handle_) {
    LOG_WARNING(
        "RuntimeCacheHandle::serialize() called before the IRuntimeCache was materialized; returning empty bytes.");
    return empty();
  }
  auto host_mem = make_trt(trt_handle_->serialize());
  if (!host_mem) {
    LOG_WARNING("IRuntimeCache::serialize() returned null host memory; returning empty bytes.");
    return empty();
  }
  auto tensor = at::empty({static_cast<int64_t>(host_mem->size())}, opts);
  std::memcpy(tensor.data_ptr(), host_mem->data(), host_mem->size());
  return tensor;
#else
  LOG_WARNING("RuntimeCacheHandle::serialize() invoked on a non-RTX build; returning empty bytes.");
  return empty();
#endif
}

void RuntimeCacheHandle::deserialize(TORCHTRT_UNUSED at::Tensor data) {
#ifdef TRT_MAJOR_RTX
  if (data.numel() == 0) {
    LOG_WARNING("RuntimeCacheHandle::deserialize() called with an empty tensor; nothing to load.");
    return;
  }
  auto contig = data.contiguous().to(at::kCPU);
  std::lock_guard<std::mutex> lock(state_mu_);
  if (trt_handle_) {
    // Live cache -- write through directly.
    trt_handle_->deserialize(contig.data_ptr(), static_cast<size_t>(contig.numel()));
  } else {
    // No live cache yet -- stash bytes for the next ``ensure_materialized``
    // call to drain. Fixes the cpp-rt warm-start path where ``wrapper.load()``
    // fires before any engine has materialized ``trt_handle_``.
    auto const* p = static_cast<uint8_t const*>(contig.data_ptr());
    pending_warm_bytes_.assign(p, p + contig.numel());
  }
#else
  LOG_WARNING("RuntimeCacheHandle::deserialize() invoked on a non-RTX build; bytes ignored.");
#endif
}

bool RuntimeCacheHandle::has_cache() const {
#ifdef TRT_MAJOR_RTX
  std::lock_guard<std::mutex> lock(state_mu_);
  return trt_handle_ != nullptr;
#else
  return false;
#endif
}

#ifdef TRT_MAJOR_RTX
std::shared_ptr<nvinfer1::IRuntimeCache> RuntimeCacheHandle::ensure_materialized(nvinfer1::IRuntimeConfig* config) {
  std::lock_guard<std::mutex> lock(state_mu_);
  if (!trt_handle_) {
    trt_handle_ = make_trt(config->createRuntimeCache());
    if (!trt_handle_) {
      return nullptr;
    }
    // Drain any bytes that ``deserialize`` stashed pre-materialization.
    // This is the cpp-rt warm-start path: ``load()`` fired before the first
    // ``ensure_materialized``, bytes parked in ``pending_warm_bytes_``,
    // first engine to ``ensure_materialized`` creates the cache and drains.
    if (!std::empty(pending_warm_bytes_)) {
      trt_handle_->deserialize(pending_warm_bytes_.data(), std::size(pending_warm_bytes_));
      pending_warm_bytes_.clear();
      pending_warm_bytes_.shrink_to_fit();
      LOG_DEBUG("Drained pending warm-start bytes into IRuntimeCache.");
    }
  }
  return trt_handle_;
}
#endif

// ---- RuntimeSettings methods ------------------------------------------------

bool RuntimeSettings::operator==(RuntimeSettings const& other) const noexcept {
  // ``runtime_cache`` compares by pointer identity: passing the same handle
  // twice through the settings setter is a no-op. Hoisted into locals because
  // ``std::tie`` requires lvalues.
  auto* this_cache = runtime_cache.get();
  auto* other_cache = other.runtime_cache.get();
  return std::tie(dynamic_shapes_kernel_specialization_strategy, cuda_graph_strategy, this_cache) ==
      std::tie(other.dynamic_shapes_kernel_specialization_strategy, other.cuda_graph_strategy, other_cache);
}

std::string RuntimeSettings::to_str() const {
  std::ostringstream os;
  os << "RuntimeSettings{" << std::endl;
  os << "  Dynamic Shapes Kernel Strategy: " << dynamic_shapes_kernel_specialization_strategy << std::endl;
  os << "  CUDA Graph Strategy: " << cuda_graph_strategy << std::endl;
  if (runtime_cache) {
    auto const& p = runtime_cache->path;
    os << "  Runtime Cache: " << (p.empty() ? "<in-memory shared>" : p) << std::endl;
  } else {
    os << "  Runtime Cache: <engine-local, in-memory>" << std::endl;
  }
  os << "}";
  return os.str();
}

std::ostream& operator<<(std::ostream& os, RuntimeSettings const& rs) {
  os << rs.to_str();
  return os;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
