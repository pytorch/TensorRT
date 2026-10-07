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
  // Shared with every other handle loaded with kSharedEnginesKey on from the same engine bytes, for
  // the same device and weight streaming request, so the weights are on the device once. The context
  // and everything below it stay this handle's own.
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
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
  // Built at init from the blob's aliased_io. Either way execute() binds the
  // aliased TRT output binding to its aliased input's caller-provided pointer,
  // so the engine's write lands in the caller's buffer; what differs is how the
  // .pte carries the buffer. Threaded: the buffer is both a delegate input arg
  // and a delegate output arg (the caller-owned mutable buffer's mutation slot),
  // and execute() reflects the result into that output EValue for ExecuTorch's
  // write-back copy_ to read. Elided -- zero-copy KV -- the buffer is an input
  // arg only, the delegate has no output for it, and execute() skips the
  // reflect: the in-place write already is the update.
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
  // kSharedActivationScratchKey was on at this handle's load. Such a context takes
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
// later call governs only the methods loaded after it, and a pooled context and
// a private-scratch one coexist in one process.
inline constexpr char kSharedActivationScratchKey[] = "use_shared_activation_scratch";

// Load-time backend option that shares one deserialized engine when the engine bytes have the same
// std::hash and size, for the same device and weight streaming request. The bytes are not compared,
// and the hash is not collision resistant: every program in the process must be trusted.
// Boolean, default true. It is a property of one load: init reads it from that load's runtime
// specs, passed to Module::load in a LoadBackendOptionsMap like the weight streaming budget, and a
// load with it false neither uses nor publishes a shared engine. Each handle still creates its own
// execution context, so handles of a shared engine can run at the same time, unless both were loaded
// with kSharedActivationScratchKey on and the engine needs activation scratch.
inline constexpr char kSharedEnginesKey[] = "use_shared_engines";

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
  //     A handle that claims pooled scratch refuses a capturing selected stream with
  //     Error::NotSupported. Other streams are not checked. Do not overlap execution with a
  //     Global capture on any thread or a ThreadLocal capture on the calling thread, even with
  //     the option off or with an engine that needs no scratch: the call can invalidate it.
  //   - cudaDeviceReset() invalidates the pool without emptying it, and the next call on that
  //     device uses what it destroyed. There is no guard: do not reset a device this backend has
  //     run a pooled engine on.
  //   - Pool failures share error codes with other failures in execute(). Error::NotSupported
  //     can also mean an output resize was refused. Error::Internal can mean handoff event
  //     creation, cudaFree(nullptr), or the final stream synchronization failed. The CUDA call
  //     can fail because of a capture or an earlier device fault. Check the error log for the cause.
  //
  // The mechanism behind all of that, the measurements, and which calls grow the pool are in the
  // backend README rather than here.
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  // Applies the runtime backend options a caller passes to
  // executorch::runtime::set_option("TensorRTBackend", ...). The only key read is
  // kSharedActivationScratchKey, a boolean. kSharedEnginesKey is load-only and is rejected here.
  ::executorch::runtime::Error set_option(
      ET_UNUSED ::executorch::runtime::BackendOptionContext& context,
      const ::executorch::runtime::Span<::executorch::runtime::BackendOption>& backend_options) override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;

 private:
  static nvinfer1::IRuntime* shared_runtime();
  friend nvinfer1::IRuntime* shared_runtime_for_testing();
};

} // namespace executorch_backend
} // namespace torch_tensorrt
