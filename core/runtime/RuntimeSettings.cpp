#include "core/runtime/RuntimeSettings.h"

#include <array>
#include <cstring>
#include <sstream>
#include <tuple>
#include <type_traits>

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

DynamicShapesKernelSpecializationStrategy to_dynamic_shapes_kernel_strategy(int64_t v) {
  // Take ``int64_t`` so out-of-range Python callers (TorchBind uses int64) are
  // caught here -- casting to int32_t first would silently wrap.
  TORCHTRT_CHECK(
      v >= 0 && static_cast<size_t>(v) < std::size(kDsStrategyNames),
      "Invalid dynamic_shapes_kernel_specialization_strategy int: " << v
                                                                    << " (expected 0..2 mapping to lazy|eager|none)");
  return static_cast<DynamicShapesKernelSpecializationStrategy>(v);
}

CudaGraphStrategy to_cuda_graph_strategy(int64_t v) {
  TORCHTRT_CHECK(
      v >= 0 && static_cast<size_t>(v) < std::size(kCgStrategyNames),
      "Invalid cuda_graph_strategy int: " << v << " (expected 0..1 mapping to disabled|whole_graph_capture)");
  return static_cast<CudaGraphStrategy>(v);
}

std::string_view ds_strategy_name(DynamicShapesKernelSpecializationStrategy v) {
  // Negative underlying values wrap to a huge ``size_t``, so a single bounds
  // check from the top covers both ends without needing ``std::clamp``.
  auto const i = static_cast<size_t>(v);
  return i < std::size(kDsStrategyNames) ? kDsStrategyNames[i] : "<unknown>";
}

std::string_view cg_strategy_name(CudaGraphStrategy v) {
  auto const i = static_cast<size_t>(v);
  return i < std::size(kCgStrategyNames) ? kCgStrategyNames[i] : "<unknown>";
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
  if (!trt_handle) {
    return empty();
  }
  auto host_mem = make_trt(trt_handle->serialize());
  if (!host_mem) {
    return empty();
  }
  auto tensor = at::empty({static_cast<int64_t>(host_mem->size())}, opts);
  std::memcpy(tensor.data_ptr(), host_mem->data(), host_mem->size());
  return tensor;
#else
  return empty();
#endif
}

void RuntimeCacheHandle::deserialize(TORCHTRT_UNUSED at::Tensor data) {
#ifdef TRT_MAJOR_RTX
  if (data.numel() == 0 || !trt_handle) {
    return;
  }
  auto contig = data.contiguous().to(at::kCPU);
  trt_handle->deserialize(contig.data_ptr(), static_cast<size_t>(contig.numel()));
#endif
}

bool RuntimeCacheHandle::has_cache() const {
#ifdef TRT_MAJOR_RTX
  return trt_handle != nullptr;
#else
  return false;
#endif
}

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
  os << "  Dynamic Shapes Kernel Strategy: " << ds_strategy_name(dynamic_shapes_kernel_specialization_strategy)
     << std::endl;
  os << "  CUDA Graph Strategy: " << cg_strategy_name(cuda_graph_strategy) << std::endl;
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
