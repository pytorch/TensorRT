/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file TensorRTBackend.h
 * @brief ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt.
 *
 * The processed blob uses the standalone TR01 wire format from
 * py/torch_tensorrt/executorch/serialization.py and is parsed directly here.
 * This runtime path intentionally does not depend on the legacy
 * Torch-TensorRT C++ runtime or libtorch.
 */
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/backend/interface.h>

#include "torch_tensorrt/executorch/OptimizationProfileSelection.h"

#include <cstdint>
#include <memory>

namespace torch_tensorrt {
namespace executorch_backend {

/// @brief Deletes TensorRT interface objects, which are freed with `delete`.
struct TRTDeleter {
  template <typename T>
  void operator()(T* p) const {
    delete p;
  }
};

/// @brief Owning pointer to a TensorRT interface object.
template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

/// @brief Forwards TensorRT diagnostics to the ExecuTorch log.
class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

/**
 * @brief The delegate ExecuTorch calls to run a TensorRT engine.
 *
 * Registered under the backend id `TensorRT`; a `.pte` produced by
 * torch_tensorrt.save(output_format="executorch") dispatches to it.
 */
class TensorRTBackend final : public ::executorch::runtime::BackendInterface {
 public:
  /// @return Whether a usable CUDA device and TensorRT runtime are present.
  bool is_available() const override;

  /// @brief Deserializes one engine from its processed blob into a handle.
  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec> compile_specs) const override;

  /**
   * @brief Binds `args` and enqueues the engine, selecting the optimization
   * profile the calling thread's OptimizationProfileGuard asked for and the
   * CUDA stream its executorch::extension::cuda::CallerStreamGuard selected.
   *
   * With a CallerStreamGuard active and no host staging required, this may
   * return while the enqueue is still in flight on the selected stream, so the
   * caller must keep device buffers alive and unmodified past return, order any
   * other stream against this one, and synchronize the stream before reading
   * device-resident outputs. The selected stream must be on the engine's device,
   * and calls on one handle must not overlap each other or its destruction.
   * Note that other CUDA delegates sharing the same guard may instead
   * synchronize before returning, so do not assume results are ready on return
   * from this one.
   */
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  /// @brief Waits for any work still in flight, then releases the handle.
  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

/**
 * @brief Selects, for the calling thread, which TensorRT optimization profile
 * the delegate runs; scope it around Module::forward() / Module::execute().
 *
 * A profile is identified by its index in the export-time profile list, so name
 * them to match whatever the exporter declared:
 *
 * @code
 * constexpr int32_t kDecodeProfile = 0;  // export order: decode first,
 * constexpr int32_t kPrefillProfile = 1; // then prefill
 *
 * executorch::extension::Module module("model.pte");
 * {
 *   OptimizationProfileGuard profile_guard(kPrefillProfile);
 *   auto result = module.forward(prefill_inputs);
 * }
 * @endcode
 *
 * The guard records a request for the current thread and does nothing else: it
 * never inspects the Module, Method, or delegate handles, and never calls
 * TensorRT. Each TensorRT delegate reads the request inside its own execute(),
 * where the engine, its lock, and the execution stream are already available,
 * and switches there. Without a guard every delegate runs profile 0.
 *
 * Composes with executorch::extension::cuda::CallerStreamGuard, which is
 * orthogonal: the stream guard says where the GPU work runs, this one says which
 * profile it runs under. A switch is issued on whichever stream execute()
 * selected.
 *
 * Contract: construct the guard on the thread that calls forward()/execute()
 * (ExecuTorch does not support concurrent execution of one Module anyway).
 * Nested guards restore the enclosing request on scope exit.
 *
 * One execution sees one consistent request, but several TensorRT engines in a
 * method apply it independently as they run. TensorRT offers no way to undo a
 * switch, so if a later engine rejects the request (a pinned index it does not
 * have, or no profile matching its inputs) it returns an error with earlier
 * engines already switched.
 *
 * @warning The index is delivered to every TensorRT delegate in the method, and
 * each one resolves it against its own profile list. Nothing makes index 1 mean
 * the same thing in two engines: if a `.pte` contains two engines compiled from
 * different profile lists, one index can select prefill in one and decode in the
 * other. Pin by index only when the engines were built from a single profile
 * list, or when the `.pte` holds one TensorRT engine. An engine with a single
 * profile is the benign case -- it runs profile 0 and logs that the pin did
 * nothing -- while a multi-profile engine that lacks the index fails the
 * execution.
 *
 * @warning Name the guard. A discarded temporary is destroyed at the end of the
 * full-expression that made it, restoring the enclosing request before
 * forward() is ever called, so the execution runs profile 0 rather than the one
 * asked for:
 *
 * @code
 * OptimizationProfileGuard(kPrefillProfile);         // no-op, guard already dead
 * OptimizationProfileGuard::automatic();             // no-op, guard already dead
 * OptimizationProfileGuard guard(kPrefillProfile);   // correct
 * auto guard = OptimizationProfileGuard::automatic(); // correct
 * @endcode
 *
 * Both mistakes are compiler warnings rather than a silently mistuned
 * execution: the pinning constructor and automatic() are each [[nodiscard]]
 * individually, because GCC applies a class-level [[nodiscard]] only to
 * returned values, not to a discarded constructor temporary.
 */
class [[nodiscard]] OptimizationProfileGuard {
 public:
  /**
   * @brief Pin an exact profile by its export-time index.
   *
   * An index this engine does not have is reported by execute(), not here, since
   * the guard never sees the engine; that is deliberate, so a computed index
   * (say -1 from a failed lookup) surfaces as an error rather than quietly
   * meaning something else.
   *
   * @param profile_index Position in the export-time profile list.
   */
  [[nodiscard]] explicit OptimizationProfileGuard(int32_t profile_index);

  /// @brief Rejected so that OptimizationProfileGuard(true) cannot become index 1.
  OptimizationProfileGuard(bool) = delete;

  /**
   * @brief Have each delegate choose from the runtime input shapes instead of
   * being told an index.
   *
   * Named rather than a sentinel index so it cannot collide with a computed one:
   *
   * @code
   * auto profile_guard = OptimizationProfileGuard::automatic();
   * @endcode
   *
   * @return A guard that must be bound to a name; discarding it selects nothing.
   */
  [[nodiscard]] static OptimizationProfileGuard automatic();

  ~OptimizationProfileGuard();
  OptimizationProfileGuard(const OptimizationProfileGuard&) = delete;
  OptimizationProfileGuard& operator=(const OptimizationProfileGuard&) = delete;

 private:
  struct AutoTag {};
  explicit OptimizationProfileGuard(AutoTag);

  ProfileRequest prev_request_;
  int32_t prev_index_;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
