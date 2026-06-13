#pragma once

#include <array>
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

// Reverse-lookup tables shared by ``to_string()`` (constexpr, inline below) and
// the ``from_underlying`` / ``from_string`` validators in the .cpp. Indices
// match the enum integer values (which mirror the nvinfer1 enums).
inline constexpr std::array<std::string_view, 3> kDsStrategyNames = {"lazy", "eager", "none"};
inline constexpr std::array<std::string_view, 2> kCgStrategyNames = {"disabled", "whole_graph_capture"};

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

// Strategy types mirroring the corresponding ``nvinfer1`` enums on TRT-RTX.
// Declared here unconditionally so non-RTX builds can still pass these values
// through the data model -- only the ``static_cast`` to the nvinfer1 type
// (inside ``TRTRuntimeConfig::ensure_initialized``) is RTX-only. Integer
// values must stay in sync with the nvinfer1 enums.
//
// Each strategy is a "newtype" class wrapping a nested ``enum Value`` so all
// string/int conversions live on the type. Pattern mirrors
// ``core/conversion/var/Var.h`` (``Var::Type`` + ``Var::type_name()``). The
// implicit ``operator Value()`` unwrap keeps the existing
// ``switch (s)`` / ``static_cast<nvinfer1::X>(s)`` / ``s == Class::kFOO``
// usage compiling unchanged.

class DynamicShapesKernelSpecializationStrategy {
 public:
  enum Value : int32_t {
    kLAZY = 0,
    kEAGER = 1,
    kNONE = 2,
  };

  constexpr DynamicShapesKernelSpecializationStrategy() = default;
  constexpr DynamicShapesKernelSpecializationStrategy(Value v) noexcept : v_(v) {}

  // Implicit unwrap to the inner enum. Lets ``switch (s)``, comparisons,
  // and ``static_cast<nvinfer1::DynamicShapesKernelSpecializationStrategy>(s)``
  // resolve through the inner ``Value`` without explicit calls.
  constexpr operator Value() const noexcept {
    return v_;
  }

  [[nodiscard]] constexpr std::string_view to_string() const noexcept {
    // Negative underlying values wrap to a huge ``size_t``, so a single
    // bounds check from the top covers both ends without needing ``std::clamp``.
    auto const i = static_cast<size_t>(v_);
    return i < std::size(kDsStrategyNames) ? kDsStrategyNames[i] : std::string_view{"<unknown>"};
  }
  [[nodiscard]] constexpr int32_t to_underlying() const noexcept {
    return static_cast<int32_t>(v_);
  }
  // ``int64_t`` so out-of-range Python callers (TorchBind uses int64) are
  // caught here. Throws on out-of-range.
  [[nodiscard]] static DynamicShapesKernelSpecializationStrategy from_underlying(int64_t v);
  // Throws on unknown name.
  [[nodiscard]] static DynamicShapesKernelSpecializationStrategy from_string(std::string_view name);

  // Two comparison overloads -- mirrors the ``TensorFormat`` pattern in
  // ``cpp/include/torch_tensorrt/torch_tensorrt.h``. Both overloads have a
  // zero-UDC viable match (one for class-vs-class, one for class-vs-Value),
  // which beats the built-in ``int == int`` (1 UDC via ``operator Value()``)
  // and the single-overload version (1 UDC via implicit Value ctor).
  constexpr bool operator==(DynamicShapesKernelSpecializationStrategy o) const noexcept {
    return v_ == o.v_;
  }
  constexpr bool operator==(Value o) const noexcept {
    return v_ == o;
  }
  constexpr bool operator!=(DynamicShapesKernelSpecializationStrategy o) const noexcept {
    return v_ != o.v_;
  }
  constexpr bool operator!=(Value o) const noexcept {
    return v_ != o;
  }
  // Disable accidental truthy use: ``if (strategy)`` would otherwise compare
  // against ``kLAZY = 0`` and silently surprise.
  explicit operator bool() = delete;

 private:
  Value v_ = kLAZY;
};

inline std::ostream& operator<<(std::ostream& os, DynamicShapesKernelSpecializationStrategy s) {
  return os << s.to_string();
}

class CudaGraphStrategy {
 public:
  enum Value : int32_t {
    kDISABLED = 0,
    kWHOLE_GRAPH_CAPTURE = 1,
  };

  constexpr CudaGraphStrategy() = default;
  constexpr CudaGraphStrategy(Value v) noexcept : v_(v) {}

  constexpr operator Value() const noexcept {
    return v_;
  }

  [[nodiscard]] constexpr std::string_view to_string() const noexcept {
    auto const i = static_cast<size_t>(v_);
    return i < std::size(kCgStrategyNames) ? kCgStrategyNames[i] : std::string_view{"<unknown>"};
  }
  [[nodiscard]] constexpr int32_t to_underlying() const noexcept {
    return static_cast<int32_t>(v_);
  }
  [[nodiscard]] static CudaGraphStrategy from_underlying(int64_t v);
  [[nodiscard]] static CudaGraphStrategy from_string(std::string_view name);

  // Two-overload comparisons (see DynamicShapesKernelSpecializationStrategy).
  constexpr bool operator==(CudaGraphStrategy o) const noexcept {
    return v_ == o.v_;
  }
  constexpr bool operator==(Value o) const noexcept {
    return v_ == o;
  }
  constexpr bool operator!=(CudaGraphStrategy o) const noexcept {
    return v_ != o.v_;
  }
  constexpr bool operator!=(Value o) const noexcept {
    return v_ != o;
  }
  explicit operator bool() = delete;

 private:
  Value v_ = kDISABLED;
};

inline std::ostream& operator<<(std::ostream& os, CudaGraphStrategy s) {
  return os << s.to_string();
}

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

std::ostream& operator<<(std::ostream& os, RuntimeSettings const& rs);

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
