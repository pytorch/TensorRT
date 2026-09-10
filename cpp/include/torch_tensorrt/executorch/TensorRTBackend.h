/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt. The processed blob uses the standalone wire format from
 * py/torch_tensorrt/executorch/serialization.py and is parsed directly here.
 * This runtime path intentionally does not depend on the legacy
 * Torch-TensorRT C++ runtime or libtorch.
 */
#pragma once

#include <NvInfer.h>

#include <executorch/runtime/backend/interface.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

struct InputProfileBounds {
  nvinfer1::Dims min{};
  nvinfer1::Dims max{};
};

struct EngineHandle {
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  TRTUniquePtr<nvinfer1::IExecutionContext> exec_ctx;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  std::vector<InputProfileBounds> input_profile_bounds;
  std::vector<void*> cached_input_ptrs;
  std::vector<size_t> cached_input_sizes;
  std::vector<void*> cached_output_ptrs;
  std::vector<size_t> cached_output_sizes;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  // Per output binding [0..num_outputs): index into input_binding_names of the
  // input it aliases (in-place KV-cache / user alias), or -1 for a normal output.
  // Built at init from the blob's aliased_io. The KV buffers are threaded by
  // ExecuTorch as caller-owned mutable-buffer delegate args (input AND aliased
  // output): execute() binds each aliased TRT output binding to its aliased
  // input's caller-provided pointer (in-place) and reflects the result into the
  // delegate output EValue, which ExecuTorch's write-back copy_ then reads.
  std::vector<int> output_aliased_input_idx;
  // Per input binding [0..num_inputs): true if any output aliases this input, so
  // its in-place (KV/user) update must land in the caller-owned storage. Built at
  // init from aliased_io; execute() uses it to reject a non-device-resident
  // aliased input instead of silently staging its update into delegate scratch.
  std::vector<bool> input_is_alias_target;
  size_t num_aliased_outputs = 0;
  int device_id = 0;
  // Whether this device can reach pageable host memory through shared host page tables, which is the
  // question the uses of this flag ask. A discrete card answers no: it can reach pageable memory, but
  // only by faulting pages in one at a time, so it takes the staged-copy path instead.
  bool pageable_host_access = false;
  // Whether exec_ctx was created kUSER_MANAGED, because
  // kSharedActivationScratchKey was on at this engine's load. Such a context takes
  // its activation scratch from the shared per-device pool on every call, unless
  // claims_pooled_scratch below is false, in which case the engine needs none and
  // never claims from the pool at all.
  bool shared_scratch = false;
  // Whether this handle takes its activation scratch from the pool. Set at init
  // only where shared_scratch above is on, and then only where the engine reports
  // needing activation scratch under some shape; a handle loaded with the option
  // off keeps this false and says nothing about what its engine needs, because it
  // has its own kSTATIC scratch either way. It is a predicate and not a size
  // because the size is never the right amount to install: enqueueV3 refuses a
  // kUSER_MANAGED context with no device memory once the engine needs any, however
  // little the shapes bound to a given call need, so execute() must hand such a
  // call some buffer -- but the engine's own figure covers every shape in the
  // profile, and installing it would size the pool for the largest of them.
  bool claims_pooled_scratch = false;
  std::mutex mu;

  ~EngineHandle();
};

// Runtime backend option that backs execution-context activation scratch with a
// shared per-device pool instead of giving every context its own. Boolean,
// default false. Read by TensorRTBackend::set_option below, and delivered as
//   executorch::runtime::set_option("TensorRTBackend", options.view())
// A context's allocation strategy is fixed when the context is created, so a
// later call governs only the engines loaded after it, and a pooled context and
// a private-scratch one coexist in one process.
inline constexpr char kSharedActivationScratchKey[] = "use_shared_activation_scratch";

class TensorRTBackend final : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override;

  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec> compile_specs) const override;

  // Runs the engine and returns once the work is finished, whatever memory it was given, so
  // outputs are readable as soon as this returns. The stream in use must be on the engine's
  // device, and calls on one handle must not overlap each other or its destruction.
  // Every buffer passed in must stay alive and unchanged until this returns. Freeing one while the
  // call runs is not detectable here or anywhere below, and returns a plausible wrong answer rather
  // than an error.
  //
  // With the shared activation scratch pool (kSharedActivationScratchKey) one buffer per device
  // backs every context created while the option was on. What a caller has to know:
  //   - Calls on two such handles on one device are serialized at submission. The backend does
  //     that itself, with a per-device lock held across the enqueue. A handle loaded with the
  //     option off keeps its own scratch, and so does one whose engine needs no activation
  //     scratch under any shape; neither is affected.
  //   - A call can wait on the host for pooled work submitted earlier on that device, and a call
  //     that grows the pool can wait for an unbounded amount of it. If the caller parks work that
  //     only the caller will release once execute() returns, the call does not return. Do not hold
  //     a pooled call behind something it has to come back to release.
  //   - Capturing a CUDA graph around this delegate is not supported, with the option on or off.
  //     With it on, a call whose selected stream is capturing is refused with
  //     Error::NotSupported. A capture on any other stream is not caught and will be invalidated.
  //   - cudaDeviceReset() invalidates the pool without emptying it, and the next call on that
  //     device uses what it destroyed. There is no guard: do not reset a device this backend has
  //     run a pooled engine on.
  //   - Only Error::Internal is the pool's own, returned when a device's handoff event cannot be
  //     created. Its other failures reuse codes execute() already returns, so the code alone does
  //     not say the pool was involved; every one of them logs at Error first.
  //
  // The mechanism behind all of that, the measurements, and which calls grow the pool are in the
  // backend README rather than here.
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  // Applies the runtime backend options a caller passes to
  // executorch::runtime::set_option("TensorRTBackend", ...). The only key read is
  // kSharedActivationScratchKey, a boolean.
  ::executorch::runtime::Error set_option(
      ET_UNUSED ::executorch::runtime::BackendOptionContext& context,
      const ::executorch::runtime::Span<::executorch::runtime::BackendOption>& backend_options) override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
