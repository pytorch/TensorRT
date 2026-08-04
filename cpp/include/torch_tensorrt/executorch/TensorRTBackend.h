/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt. The processed blob uses the standalone TR01 wire format from
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

struct TRTDeleter {
  template <typename T>
  void operator()(T* p) const {
    delete p;
  }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTBackend final : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override;

  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec> compile_specs) const override;

  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

// Selects, for the calling thread, the CUDA stream the delegate runs TensorRT on;
// scope it around execution.
//
// Confines inference to a CUDA green context's SM partition when the caller
// passes a cuGreenCtxStreamCreate stream: confinement rides the stream (the green
// context need not be current), and cudaStreamPerThread — the no-guard default —
// is rejected while a green context is current. While active, device-resident
// outputs are left enqueued on the stream (no end sync) to compose with later GPU
// work.
//
// Contract: the stream is on the engine's device and outlives the guard; a handle
// is executed by one thread at a time. On the no-end-sync path (guard active, all
// I/O device-resident) execute() returns with the TensorRT enqueue still in flight
// on the stream; the delegate itself orders the next execute() on, and the
// destruction of, that handle after the work completes (via an internal completion
// event), so the caller need only synchronize the stream before reading
// device-resident outputs.
class CudaStreamGuard {
 public:
  explicit CudaStreamGuard(cudaStream_t stream);
  ~CudaStreamGuard();
  CudaStreamGuard(const CudaStreamGuard&) = delete;
  CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;

 private:
  cudaStream_t prev_stream_;
  bool prev_set_;
};

// Selects, for the calling thread, which TensorRT optimization profile the
// delegate runs; scope it around Module::forward() / Module::execute(). A
// profile is identified by its index in the export-time profile list, so name
// them to match whatever the exporter declared:
//
//   constexpr int32_t kDecodeProfile = 0;  // export order: decode first,
//   constexpr int32_t kPrefillProfile = 1; // then prefill
//
//   executorch::extension::Module module("model.pte");
//   {
//     OptimizationProfileGuard profile_guard(kPrefillProfile);
//     auto result = module.forward(prefill_inputs);
//   }
//
// The guard records a request for the current thread and does nothing else: it
// never inspects the Module, Method, or delegate handles, and never calls
// TensorRT. Each TensorRT delegate reads the request inside its own execute(),
// where the engine, its lock, and the execution stream are already available,
// and switches there. Without a guard every delegate runs profile 0.
//
// Composes with CudaStreamGuard, which is orthogonal: the stream guard says
// where the GPU work runs, this one says which profile it runs under. A switch
// is issued on whichever stream execute() selected.
//
// Contract: construct the guard on the thread that calls forward()/execute()
// (ExecuTorch does not support concurrent execution of one Module anyway).
// Nested guards restore the enclosing request on scope exit.
//
// One execution sees one consistent request, but several TensorRT engines in a
// method apply it independently as they run. TensorRT offers no way to undo a
// switch, so if a later engine rejects the request (a pinned index it does not
// have, or no profile matching its inputs) it returns an error with earlier
// engines already switched.
//
// The index is delivered to every TensorRT delegate in the method, and each one
// resolves it against its own profile list. Nothing makes index 1 mean the same
// thing in two engines: if a .pte contains two engines compiled from different
// profile lists, one index can select prefill in one and decode in the other.
// Pin by index only when the engines were built from a single profile list, or
// when the .pte holds one TensorRT engine. An engine with a single profile is
// the benign case -- it runs profile 0 and logs that the pin did nothing --
// while a multi-profile engine that lacks the index fails the execution.
class OptimizationProfileGuard {
 public:
  // Pin an exact profile by its export-time index. An index this engine does not
  // have is reported by execute(), not here, since the guard never sees the
  // engine; that is deliberate, so a computed index (say -1 from a failed
  // lookup) surfaces as an error rather than quietly meaning something else.
  explicit OptimizationProfileGuard(int32_t profile_index);

  // Reject bool so that OptimizationProfileGuard(true) cannot become index 1.
  OptimizationProfileGuard(bool) = delete;

  // Have each delegate choose from the runtime input shapes instead of being
  // told an index. Named rather than a sentinel index so it cannot collide with
  // a computed one:
  //
  //   auto profile_guard = OptimizationProfileGuard::automatic();
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
