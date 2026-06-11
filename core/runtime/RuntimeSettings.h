#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "ATen/core/Tensor.h"
#include "ATen/core/ivalue.h"
#include "NvInfer.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

// A passive wrapper around an ``IRuntimeCache``. Registered as a torchbind class
// so it can be passed by ``c10::intrusive_ptr`` across the Python/C++ boundary;
// the same handle gives both runtimes the same underlying ``IRuntimeCache*``.
//
// File I/O lives on the Python side (filelock + on-disk persistence via
// the ``serialize`` / ``deserialize`` members below). The C++ struct is purely
// a holder; ``path`` is informational and is not consulted by the C++ runtime.
class RuntimeCacheHandle : public torch::CustomClassHolder {
 public:
  std::string path;

  explicit RuntimeCacheHandle(std::string p = "") : path(std::move(p)) {}

  // Expose the underlying ``IRuntimeCache`` bytes for the Python side to persist
  // under filelock. Returns an empty uint8 tensor when no cache is attached, or
  // on non-RTX builds.
  //
  // ``at::Tensor`` is used (rather than ``std::string``) because TorchBind
  // forces ``std::string`` to round-trip through Python ``str`` (UTF-8), and
  // serialized cache bytes are not valid UTF-8.
  [[nodiscard]] at::Tensor serialize() const;

  // Inverse of ``serialize``. If the underlying ``IRuntimeCache`` is live,
  // deserializes directly; otherwise stashes the bytes for the next
  // ``ensure_materialized`` call to drain. No-op on empty input or non-RTX
  // builds.
  void deserialize(at::Tensor data);

  // True iff an engine has populated the underlying ``IRuntimeCache``.
  // Always false on non-RTX builds.
  [[nodiscard]] bool has_cache() const;

#ifdef TRT_MAJOR_RTX
  // Idempotently materialize the underlying ``IRuntimeCache`` via ``config``
  // and drain any bytes that ``deserialize`` stashed before materialization.
  // Returns the now-live cache (or ``nullptr`` if creation failed).
  //
  // Safe to call concurrently from multiple engines sharing this handle
  // (which is what ``runtime_cache([a, b], path)`` does): exactly one caller
  // creates + drains, others see the already-live cache.
  std::shared_ptr<nvinfer1::IRuntimeCache> ensure_materialized(nvinfer1::IRuntimeConfig* config);

 private:
  // All cache-state access is serialized through ``state_mu_``: this handle
  // is potentially shared between engines and may be touched from Python (GIL
  // does not extend into the cpp deserialize path).
  // Invariant: ``trt_handle_`` non-null XOR ``pending_warm_bytes_`` may carry
  // bytes; both fields require ``state_mu_`` to read or write.
  mutable std::mutex state_mu_;
  std::shared_ptr<nvinfer1::IRuntimeCache> trt_handle_;
  std::vector<uint8_t> pending_warm_bytes_;
#endif
};

// Strategy enums mirroring the corresponding ``nvinfer1`` enums on TRT-RTX.
// Declared here unconditionally so non-RTX builds can still pass these values
// through the data model -- only the ``static_cast`` to the nvinfer1 type
// (inside ``TRTRuntimeConfig::ensure_initialized``) is RTX-only. Integer
// values must stay in sync with the nvinfer1 enums.
enum class DynamicShapesKernelSpecializationStrategy : int32_t {
  kLAZY = 0,
  kEAGER = 1,
  kNONE = 2,
};

enum class CudaGraphStrategy : int32_t {
  kDISABLED = 0,
  kWHOLE_GRAPH_CAPTURE = 1,
};

// Boundary validators: take the int that crossed the Py->C++ wire and return
// the enum (or throw with a clear message on out-of-range).
[[nodiscard]] DynamicShapesKernelSpecializationStrategy to_dynamic_shapes_kernel_strategy(int64_t v);
[[nodiscard]] CudaGraphStrategy to_cuda_graph_strategy(int64_t v);

// Per-engine runtime-only knobs that move across the Python/C++ boundary.
struct RuntimeSettings {
  DynamicShapesKernelSpecializationStrategy dynamic_shapes_kernel_specialization_strategy =
      DynamicShapesKernelSpecializationStrategy::kLAZY;
  CudaGraphStrategy cuda_graph_strategy = CudaGraphStrategy::kDISABLED;
  c10::intrusive_ptr<RuntimeCacheHandle> runtime_cache = nullptr;

  bool operator==(RuntimeSettings const& other) const noexcept;
  bool operator!=(RuntimeSettings const& other) const noexcept {
    return !(*this == other);
  }

  [[nodiscard]] std::string to_str() const;
};

// Reverse-lookup helpers (enum -> name); out-of-range renders as ``"<unknown>"``.
[[nodiscard]] std::string_view ds_strategy_name(DynamicShapesKernelSpecializationStrategy v);
[[nodiscard]] std::string_view cg_strategy_name(CudaGraphStrategy v);

std::ostream& operator<<(std::ostream& os, RuntimeSettings const& rs);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
