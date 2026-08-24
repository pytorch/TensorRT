/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
 * @brief Runtime backend option that backs execution-context activation scratch
 * with a shared per-device pool instead of giving every context its own.
 *
 * Boolean, default false. Read by TensorRTBackend::set_option, and delivered as
 * `executorch::runtime::set_option("TensorRTBackend", options.view())`. A
 * context's allocation strategy is fixed when the context is created, so a
 * later call governs only the methods loaded after it, and a pooled context and
 * a private-scratch one coexist in one process.
 */
inline constexpr char kSharedActivationScratchKey[] = "use_shared_activation_scratch";

/**
 * @brief Load-time backend option that shares one deserialized engine when the
 * engine bytes have the same std::hash and size, for the same device and weight
 * streaming request.
 *
 * The bytes are not compared, and the hash is not collision resistant: every
 * program in the process must be trusted. Boolean, default true. It is a
 * property of one load: init reads it from that load's runtime specs, passed to
 * Module::load in a LoadBackendOptionsMap like the weight streaming budget, and
 * a load with it false neither uses nor publishes a shared engine. Each handle
 * still creates its own execution context, so handles of a shared engine can run
 * at the same time, unless both were loaded with kSharedActivationScratchKey on
 * and the engine needs activation scratch.
 */
inline constexpr char kSharedEnginesKey[] = "use_shared_engines";

/**
 * @brief Records each eligible engine as a CUDA graph and replays it.
 *
 * Read when an engine loads, from the load option of this name, a boolean passed
 * to Module::load, which overrides everything else. Otherwise an explicit process
 * option of false refuses recording. Then the program's compile spec, b"1" or
 * b"0", overrides the process default, which is false unless set. Any other
 * compile spec value, or the key twice, fails the load. Engines with pooled
 * scratch or aliased outputs, GPUs without stream-ordered memory, and drivers
 * older than CUDA 12.5 keep ordinary enqueueV3. Caller streams in the current
 * ordinary context can replay, as can a call with no caller stream. Green
 * context and other context streams take the plain path and keep the graph.
 */
inline constexpr char kCudaGraphsKey[] = "use_cuda_graphs";

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
   * @brief Binds `args` and runs the engine, selecting the optimization profile
   * the calling thread's OptimizationProfileGuard asked for, on the CUDA stream
   * its executorch::extension::cuda::CallerStreamGuard selected.
   *
   * Returns once the work is finished, whatever memory it was given, so outputs
   * are readable as soon as this returns. The stream in use must be on the
   * engine's device, and calls on one handle must not overlap each other or its
   * destruction. Every buffer passed in must stay alive and unchanged until this
   * returns. Freeing one while the call runs is not detectable here or anywhere
   * below, and returns a plausible wrong answer rather than an error.
   *
   * With the shared activation scratch pool (kSharedActivationScratchKey) one
   * buffer per device backs every context created while the option was on. What
   * a caller has to know:
   *   - Calls on two such handles on one device are serialized at submission.
   *     The backend does that itself, with a per-device lock held across the
   *     enqueue. A handle loaded with the option off keeps its own scratch, and
   *     so does one whose engine needs no activation scratch under any shape;
   *     neither is affected.
   *   - A call can wait on the host for pooled work submitted earlier on that
   *     device, and a call that grows the pool can wait for an unbounded amount
   *     of it. If the caller parks work that only the caller will release once
   *     execute() returns, the call does not return. Do not hold a pooled call
   *     behind something it has to come back to release.
   *   - Capturing a CUDA graph around this delegate is not supported, with the
   *     option on or off. A handle that claims pooled scratch refuses a
   *     capturing selected stream with Error::NotSupported. Other streams are
   *     not checked. Do not overlap execution with a Global capture on any
   *     thread or a ThreadLocal capture on the calling thread, even with the
   *     option off or with an engine that needs no scratch: the call can
   *     invalidate it.
   *   - cudaDeviceReset() invalidates the pool without emptying it, and the next
   *     call on that device uses what it destroyed. There is no guard: do not
   *     reset a device this backend has run a pooled engine on.
   *   - Pool failures share error codes with other failures in execute().
   *     Error::NotSupported can also mean an output resize was refused.
   *     Error::Internal can mean handoff event creation, cudaFree(nullptr), or
   *     the final stream synchronization failed. The CUDA call can fail because
   *     of a capture or an earlier device fault. Check the error log for the
   *     cause.
   *
   * The mechanism behind all of that, the measurements, and which calls grow
   * the pool are in the backend README rather than here.
   */
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  /**
   * @brief Applies the runtime backend options a caller passes to
   * `executorch::runtime::set_option("TensorRTBackend", ...)`.
   *
   * The keys read are kSharedActivationScratchKey and kCudaGraphsKey, both
   * booleans. A kCudaGraphsKey load option wins over this one, and an explicit
   * false refuses a saved true. kSharedEnginesKey is load-only and is rejected
   * here.
   */
  ::executorch::runtime::Error set_option(
      ET_UNUSED ::executorch::runtime::BackendOptionContext& context,
      const ::executorch::runtime::Span<::executorch::runtime::BackendOption>& backend_options) override;

  /// @brief Releases the handle; execute() already waited for its work.
  void destroy(::executorch::runtime::DelegateHandle* handle) const override;

 private:
  static nvinfer1::IRuntime* shared_runtime();
  friend nvinfer1::IRuntime* shared_runtime_for_testing();
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
 */
class OptimizationProfileGuard {
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
  explicit OptimizationProfileGuard(int32_t profile_index);

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
   */
  static OptimizationProfileGuard automatic();

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
