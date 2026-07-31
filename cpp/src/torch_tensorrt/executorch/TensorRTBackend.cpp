/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_tensorrt/executorch/TensorRTBackend.h"
#include "torch_tensorrt/executorch/TensorRTBindingNames.h"
#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

namespace torch_tensorrt {
namespace executorch_backend {

using ::executorch::aten::SizesType;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

#define TORCHTRT_ET_CHECK_NOT_NULL(VALUE, ERROR_CODE, ...) \
  do {                                                     \
    if ((VALUE) == nullptr) {                              \
      ET_LOG(Error, __VA_ARGS__);                          \
      return ERROR_CODE;                                   \
    }                                                      \
  } while (false)

namespace {
thread_local cudaStream_t g_user_stream = nullptr;
thread_local bool g_user_stream_set = false;
} // namespace

CudaStreamGuard::CudaStreamGuard(cudaStream_t stream) : prev_stream_(g_user_stream), prev_set_(g_user_stream_set) {
  g_user_stream = stream;
  g_user_stream_set = true;
}

CudaStreamGuard::~CudaStreamGuard() {
  g_user_stream = prev_stream_;
  g_user_stream_set = prev_set_;
}

void TRTLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kERROR) {
    ET_LOG(Error, "TensorRT: %s", msg);
  } else if (severity == Severity::kWARNING) {
    ET_LOG(Info, "TensorRT warning: %s", msg);
  }
}

EngineHandle::~EngineHandle() {
  cudaSetDevice(device_id);
  // A fast-path execute() may have returned with its enqueue still in flight on the
  // caller's stream, still using exec_ctx and the cached staging buffers. Wait on
  // the recorded completion event before destroying the context or freeing the
  // buffers. We wait on the event, not the stream, so this stays valid even if the
  // caller already destroyed the stream. Non-skip executes synchronized inline, so
  // inflight_pending is false there. Fall back to a device sync if no event exists.
  if (inflight_event != nullptr) {
    if (inflight_pending) {
      cudaError_t err = cudaEventSynchronize(inflight_event);
      if (err != cudaSuccess) {
        ET_LOG(Error, "EngineHandle::~EngineHandle: cudaEventSynchronize failed: %s", cudaGetErrorString(err));
        cudaGetLastError(); // clear sticky error; tear down regardless
      }
      inflight_pending = false;
    }
  } else {
    cudaDeviceSynchronize();
  }
  for (void* p : cached_input_ptrs) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  for (void* p : cached_output_ptrs) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  exec_ctx.reset();
  engine.reset();
  runtime.reset();
  if (inflight_event != nullptr) {
    cudaEventDestroy(inflight_event);
    inflight_event = nullptr;
  }
}

namespace {

struct EngineHandleDeleter {
  void operator()(EngineHandle* handle) const {
    if (handle != nullptr) {
      handle->~EngineHandle();
    }
  }
};

nvinfer1::Dims to_trt_dims(const exec_aten::Tensor& t) {
  nvinfer1::Dims dims{};
  dims.nbDims = t.dim();
  if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
    return dims;
  }
  for (int d = 0; d < t.dim(); ++d) {
    dims.d[d] = static_cast<int64_t>(t.size(d));
  }
  return dims;
}

bool infer_binding_names(
    nvinfer1::ICudaEngine* engine,
    std::vector<std::string>& inputs,
    std::vector<std::string>& outputs) {
  if (engine == nullptr) {
    return false;
  }

  detail::TensorRTBindingNames binding_names;
  if (!detail::infer_engine_binding_names(*engine, binding_names)) {
    return false;
  }

  inputs = std::move(binding_names.input_names);
  outputs = std::move(binding_names.output_names);
  return true;
}

Error initialize_engine_io(EngineHandle& handle) {
  if (handle.input_binding_names.empty() && handle.output_binding_names.empty() &&
      !infer_binding_names(handle.engine.get(), handle.input_binding_names, handle.output_binding_names)) {
    ET_LOG(Error, "TensorRTBackend::init: failed to infer TensorRT binding names");
    return Error::InvalidProgram;
  }

  handle.num_inputs = handle.input_binding_names.size();
  handle.num_outputs = handle.output_binding_names.size();

  handle.exec_ctx.reset(handle.engine->createExecutionContext());
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle.exec_ctx, Error::InvalidProgram, "TensorRTBackend::init: failed to create TensorRT execution context");

  return Error::Ok;
}

Error initialize_input_profiles(EngineHandle& handle) {
  for (const auto& name : handle.input_binding_names) {
    if (handle.engine->isShapeInferenceIO(name.c_str())) {
      ET_LOG(Error, "TensorRTBackend::init: shape tensor input '%s' is not supported", name.c_str());
      return Error::InvalidProgram;
    }
  }

  handle.input_profile_bounds.reserve(handle.num_inputs);
  for (const auto& name : handle.input_binding_names) {
    InputProfileBounds bounds;
    bounds.min = handle.engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
    bounds.max = handle.engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    if (bounds.min.nbDims < 0 || bounds.max.nbDims < 0) {
      ET_LOG(Error, "TensorRTBackend::init: getProfileShape failed for input '%s'", name.c_str());
      return Error::InvalidProgram;
    }
    handle.input_profile_bounds.push_back(bounds);
  }

  return Error::Ok;
}

bool is_cuda_accessible_ptr(const void* ptr) {
  if (ptr == nullptr) {
    return false;
  }
  cudaPointerAttributes attrs{};
  const cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
}

} // namespace

// ---------------------------------------------------------------------------
// is_available
// ---------------------------------------------------------------------------
bool TensorRTBackend::is_available() const {
  TRTLogger logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  return runtime != nullptr;
}

// ---------------------------------------------------------------------------
// init
//
// Deserializes the processed blob into a TensorRT engine handle. The handle is
// placement-new'd into memory provided by the ExecuTorch MemoryAllocator so
// that ExecuTorch owns the arena lifetime; destroy() calls the destructor.
// ---------------------------------------------------------------------------
Result<DelegateHandle*> TensorRTBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;

  TORCHTRT_ET_CHECK_NOT_NULL(processed, Error::InvalidArgument, "TensorRTBackend::init: null processed buffer");
  TORCHTRT_ET_CHECK_NOT_NULL(processed->data(), Error::InvalidArgument, "TensorRTBackend::init: null processed buffer");

  TensorRTBlobHeader header;
  if (!TensorRTBlobHeader::parse(processed->data(), processed->size(), header)) {
    ET_LOG(Error, "TensorRTBackend::init: failed to parse TR01 TensorRT blob");
    return Error::InvalidProgram;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  TORCHTRT_ET_CHECK_NOT_NULL(allocator, Error::InvalidState, "TensorRTBackend::init: null runtime allocator");

  EngineHandle* handle = allocator->allocateInstance<EngineHandle>();
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle, Error::MemoryAllocationFailed, "TensorRTBackend::init: EngineHandle allocation failed");
  new (handle) EngineHandle();
  std::unique_ptr<EngineHandle, EngineHandleDeleter> handle_guard(handle);

  handle->input_binding_names = std::move(header.input_binding_names);
  handle->output_binding_names = std::move(header.output_binding_names);
  handle->device_id = header.device_id;

  cudaError_t cuda_err = cudaSetDevice(handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error, "TensorRTBackend::init: cudaSetDevice(%d) failed: %s", handle->device_id, cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }

  // Created while device_id is current so the event belongs to the engine's device.
  // It orders a later execute()/teardown after a skip-sync enqueue (see execute()
  // and ~EngineHandle). Blocking-sync so the host yields instead of busy-spinning.
  cuda_err = cudaEventCreateWithFlags(&handle->inflight_event, cudaEventDisableTiming | cudaEventBlockingSync);
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "TensorRTBackend::init: cudaEventCreateWithFlags failed: %s", cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }

  int is_integrated = 0;
  cuda_err = cudaDeviceGetAttribute(&is_integrated, cudaDevAttrIntegrated, handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Info,
        "TensorRTBackend::init: cudaDeviceGetAttribute(cudaDevAttrIntegrated) failed: %s",
        cudaGetErrorString(cuda_err));
  }
  handle->unified_memory = is_integrated != 0;

  handle->runtime.reset(nvinfer1::createInferRuntime(handle->logger));
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle->runtime, Error::InvalidProgram, "TensorRTBackend::init: failed to create TensorRT runtime");

  const void* engine_data = TensorRTBlobHeader::engine_data(processed->data(), header);
  handle->engine.reset(handle->runtime->deserializeCudaEngine(engine_data, header.engine_size));
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle->engine, Error::InvalidProgram, "TensorRTBackend::init: failed to deserialize TensorRT engine");

  Error err = initialize_engine_io(*handle);
  if (err != Error::Ok) {
    return err;
  }

  // Map each aliased output binding to the index of the input it aliases so
  // execute() can bind it to that input's device pointer (in-place).
  // Non-aliased models have an empty header.aliased_io -> all -1, unchanged path.
  handle->output_aliased_input_idx.assign(handle->num_outputs, -1);
  handle->input_is_alias_target.assign(handle->num_inputs, false);
  for (const auto& ab : header.aliased_io) {
    int oi = -1;
    for (size_t k = 0; k < handle->output_binding_names.size(); ++k) {
      if (handle->output_binding_names[k] == ab.output) {
        oi = static_cast<int>(k);
        break;
      }
    }
    int ii = -1;
    for (size_t k = 0; k < handle->input_binding_names.size(); ++k) {
      if (handle->input_binding_names[k] == ab.input) {
        ii = static_cast<int>(k);
        break;
      }
    }
    if (oi < 0 || ii < 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::init: aliased_io names not found (output='%s', input='%s')",
          ab.output.c_str(),
          ab.input.c_str());
      return Error::InvalidProgram;
    }
    // Validate the alias kind against the two we understand. The blob parser
    // defaults a missing "kind" to "kv_cache_update"; any other value is a
    // corrupt or newer-than-us wire format we can't safely bind, so fail loudly
    // rather than fall through and treat it as a KV alias (which would bind two
    // tensors to the same storage). Mirrors the Python _reconcile_aliased_io.
    if (ab.kind != "kv_cache_update" && ab.kind != "user") {
      ET_LOG(
          Error,
          "TensorRTBackend::init: aliased_io entry (output='%s') has unknown kind '%s'",
          ab.output.c_str(),
          ab.kind.c_str());
      return Error::InvalidProgram;
    }
    if (ab.kind == "kv_cache_update") {
      // TensorRT's IKVCacheUpdateLayer aliasing is the source of truth for
      // kv_cache_update; the persisted map must agree with what the engine
      // reports, else the blob is inconsistent with its own engine.
      const char* trt_alias = handle->engine->getAliasedInputTensor(ab.output.c_str());
      if (trt_alias == nullptr || ab.input != trt_alias) {
        ET_LOG(
            Error,
            "TensorRTBackend::init: kv_cache_update alias for output '%s' disagrees with the "
            "engine (persisted input='%s', engine input='%s')",
            ab.output.c_str(),
            ab.input.c_str(),
            trt_alias == nullptr ? "<none>" : trt_alias);
        return Error::InvalidProgram;
      }
    } else {
      // AliasKind::USER aliases are declared by Torch-TensorRT and not tracked
      // by TensorRT, so it can't validate them; confirm the aliased output and
      // input share a shape before binding them to the same storage.
      const nvinfer1::Dims od = handle->engine->getTensorShape(ab.output.c_str());
      const nvinfer1::Dims id = handle->engine->getTensorShape(ab.input.c_str());
      bool compatible = od.nbDims == id.nbDims;
      for (int d = 0; compatible && d < od.nbDims; ++d) {
        compatible = od.d[d] == id.d[d];
      }
      if (!compatible) {
        ET_LOG(
            Error,
            "TensorRTBackend::init: user alias output '%s' shape is incompatible with input '%s'",
            ab.output.c_str(),
            ab.input.c_str());
        return Error::InvalidProgram;
      }
    }
    handle->output_aliased_input_idx[static_cast<size_t>(oi)] = ii;
    handle->input_is_alias_target[static_cast<size_t>(ii)] = true;
    ++handle->num_aliased_outputs;
  }

  if (handle->num_aliased_outputs > 0) {
    ET_LOG(
        Info,
        "TensorRTBackend::init: %zu aliased output(s) bound in-place to caller-owned inputs",
        handle->num_aliased_outputs);
  }

  err = initialize_input_profiles(*handle);
  if (err != Error::Ok) {
    return err;
  }

  processed->Free();

  ET_LOG(
      Info,
      "TensorRTBackend::init: TensorRT engine ready (%zu inputs, %zu outputs)",
      handle->num_inputs,
      handle->num_outputs);

  handle_guard.release();
  return static_cast<DelegateHandle*>(handle);
}

// ---------------------------------------------------------------------------
// execute
//
// Binds the ExecuTorch input/output tensor data pointers directly to the
// TRT IExecutionContext and calls enqueueV3().  ExecuTorch pre-allocates
// all output tensors before calling execute(), so we only need to register
// their addresses; no separate output allocation is required.
//
// Args layout (mirroring the Python exporter):
//   args[0 .. num_inputs-1]             – input EValues
//   args[num_inputs .. num_inputs+num_outputs-1] – output EValues
// ---------------------------------------------------------------------------
Error TensorRTBackend::execute(BackendExecutionContext& context, DelegateHandle* handle, Span<EValue*> args) const {
  (void)context;
  TORCHTRT_ET_CHECK_NOT_NULL(handle, Error::InvalidArgument, "TensorRTBackend::execute: null delegate handle");
  auto* engine = static_cast<EngineHandle*>(handle);

  const size_t num_inputs = engine->num_inputs;
  const size_t num_outputs = engine->num_outputs;
  // Caller-owned KV: every input is a delegate arg, and each aliased output is
  // threaded as a delegate output arg (the caller-owned mutable buffer's mutation
  // slot), so all engine bindings map 1:1 to delegate args.
  const size_t num_delegate_outputs = num_outputs;
  const size_t num_delegate_inputs = num_inputs;
  if (args.size() < num_delegate_inputs + num_delegate_outputs) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: expected at least %zu args, got %zu",
        num_delegate_inputs + num_delegate_outputs,
        args.size());
    return Error::InvalidArgument;
  }

  int entry_device = -1;
  cudaError_t cuda_err = cudaGetDevice(&entry_device);
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "TensorRTBackend::execute: cudaGetDevice failed: %s", cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }
  // Put the engine on its own device for multi-GPU correctness, restoring the
  // caller's device on exit; green-context confinement rides the selected stream,
  // independent of the current device/context.
  const bool switch_device = (entry_device != engine->device_id);
  if (switch_device) {
    cuda_err = cudaSetDevice(engine->device_id);
    if (cuda_err != cudaSuccess) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: cudaSetDevice(%d) failed: %s",
          engine->device_id,
          cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
  }
  struct DeviceRestore {
    int device;
    bool active;
    ~DeviceRestore() {
      if (active) {
        cudaSetDevice(device);
      }
    }
  } device_restore{entry_device, switch_device};

  std::unique_lock<std::mutex> lock(engine->mu);

  nvinfer1::IExecutionContext* ctx = engine->exec_ctx.get();
  TORCHTRT_ET_CHECK_NOT_NULL(ctx, Error::InvalidState, "TensorRTBackend::execute: backend is not initialized");

  // A prior fast-path execute() may have returned with its enqueue still in flight
  // on the shared exec_ctx. Wait for it before reconfiguring the context below:
  // TensorRT forbids mutating a context while one of its enqueues is in flight, and
  // setInputShape/setTensorAddress run on the host, so this must be a host-side wait.
  if (engine->inflight_pending) {
    cuda_err = cudaEventSynchronize(engine->inflight_event);
    engine->inflight_pending = false;
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "TensorRTBackend::execute: cudaEventSynchronize failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
  }
  cudaStream_t stream = g_user_stream_set ? g_user_stream : cudaStreamPerThread;
  bool output_staged_to_host = false;
  bool input_staged_from_host = false;

  if (engine->cached_input_ptrs.empty()) {
    engine->cached_input_ptrs.resize(num_inputs, nullptr);
    engine->cached_input_sizes.resize(num_inputs, 0);
  }
  if (engine->cached_output_ptrs.empty()) {
    engine->cached_output_ptrs.resize(num_outputs, nullptr);
    engine->cached_output_sizes.resize(num_outputs, 0);
  }

  // ------------------------------------------------------------------
  // 1. Bind input shapes and addresses
  // ------------------------------------------------------------------
  // Device pointer each input binding was bound to; aliased outputs reuse the
  // pointer of the input they alias so their update lands in-place.
  std::vector<void*> input_bind_ptrs(num_inputs, nullptr);
  size_t arg_idx = 0; // running index into delegate args
  for (size_t i = 0; i < num_inputs; ++i) {
    const std::string& name = engine->input_binding_names[i];

    EValue* arg = args[arg_idx++];
    TORCHTRT_ET_CHECK_NOT_NULL(
        arg, Error::InvalidArgument, "TensorRTBackend::execute: input arg %zu is not a tensor", i);
    if (!arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: input %zu is not a tensor", i);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_in = arg->toTensor();
    nvinfer1::Dims dims = to_trt_dims(et_in);
    if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
      ET_LOG(Error, "TensorRTBackend::execute: input '%s' rank exceeds TensorRT limit", name.c_str());
      return Error::InvalidArgument;
    }

    const auto& bounds = engine->input_profile_bounds[i];
    if (dims.nbDims != bounds.min.nbDims) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: input '%s' rank %d does not match profile rank %d",
          name.c_str(),
          dims.nbDims,
          bounds.min.nbDims);
      return Error::InvalidArgument;
    }
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < bounds.min.d[d] || dims.d[d] > bounds.max.d[d]) {
        ET_LOG(Error, "TensorRTBackend::execute: input '%s' dim %d is outside profile bounds", name.c_str(), d);
        return Error::InvalidArgument;
      }
    }

    if (!ctx->setInputShape(name.c_str(), dims)) {
      ET_LOG(Error, "TensorRTBackend::execute: setInputShape failed for '%s'", name.c_str());
      return Error::InvalidState;
    }

    // Caller-owned aliased input: an aliased output binds in-place to this
    // input's device pointer, so its update must land in the caller's storage.
    // If it isn't device-resident the branches below would stage it through
    // delegate scratch, and the in-place update (bound to that scratch) would be
    // silently lost on the next execute() when the staging copy re-reads the
    // caller's unchanged buffer. Fail loudly instead.
    if (engine->input_is_alias_target[i]) {
      const bool device_resident =
          et_in.nbytes() > 0 && (engine->unified_memory || is_cuda_accessible_ptr(et_in.const_data_ptr()));
      if (!device_resident) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: aliased input '%s' must be device-resident (non-empty and "
            "CUDA-accessible or unified memory); its caller-owned in-place update cannot be staged "
            "through host scratch",
            name.c_str());
        return Error::InvalidArgument;
      }
    }

    void* bind_ptr = nullptr;
    if (et_in.nbytes() == 0) {
      if (engine->cached_input_sizes[i] == 0) {
        cuda_err = cudaMalloc(&engine->cached_input_ptrs[i], 1);
        if (cuda_err != cudaSuccess) {
          return Error::MemoryAllocationFailed;
        }
        engine->cached_input_sizes[i] = 1;
      }
      bind_ptr = engine->cached_input_ptrs[i];
    } else if (engine->unified_memory || is_cuda_accessible_ptr(et_in.const_data_ptr())) {
      bind_ptr = et_in.mutable_data_ptr();
    } else {
      const size_t needed = et_in.nbytes();
      if (needed > engine->cached_input_sizes[i]) {
        if (engine->cached_input_ptrs[i] != nullptr) {
          cudaFree(engine->cached_input_ptrs[i]);
        }
        cuda_err = cudaMalloc(&engine->cached_input_ptrs[i], needed);
        if (cuda_err != cudaSuccess) {
          engine->cached_input_ptrs[i] = nullptr;
          engine->cached_input_sizes[i] = 0;
          return Error::MemoryAllocationFailed;
        }
        engine->cached_input_sizes[i] = needed;
      }
      bind_ptr = engine->cached_input_ptrs[i];
      input_staged_from_host = true;
      cuda_err = cudaMemcpyAsync(bind_ptr, et_in.const_data_ptr(), needed, cudaMemcpyHostToDevice, stream);
      if (cuda_err != cudaSuccess) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: H2D copy failed for input '%s': %s",
            name.c_str(),
            cudaGetErrorString(cuda_err));
        return Error::InvalidProgram;
      }
    }

    input_bind_ptrs[i] = bind_ptr;
    if (!ctx->setTensorAddress(name.c_str(), bind_ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for input '%s'", name.c_str());
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 2. Infer output shapes (requires all input shapes to be set first)
  // ------------------------------------------------------------------
  {
    const int32_t io_size = engine->engine->getNbIOTensors();
    std::vector<const char*> unresolved(static_cast<size_t>(io_size), nullptr);
    const int32_t n_unresolved = ctx->inferShapes(io_size, unresolved.data());
    if (n_unresolved != 0) {
      ET_LOG(Error, "TensorRTBackend::execute: inferShapes could not resolve %d tensor(s)", n_unresolved);
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 3. Bind output addresses
  // ExecuTorch pre-allocates output tensors at the maximum shape for
  // dynamic models.  After inferShapes() TRT knows the actual output
  // dims, so update the ExecuTorch TensorImpl's sizes before computing
  // nbytes() and before the Python binding reads back the shape.
  // If the buffer is CPU, stage through a temporary CUDA allocation.
  // ------------------------------------------------------------------
  // (arg index, device_src ptr) for outputs staged through a device buffer.
  std::vector<std::pair<size_t, void*>> outputs_needing_copy;
  // Caller-owned KV: (dst = delegate output EValue ptr, src = aliased input ptr,
  // nbytes). The engine updates the aliased input in place; reflect that into the
  // delegate output EValue after enqueue so ExecuTorch's write-back copy_ sees the
  // updated cache. Skipped when dst == src (memory planner aliased them: zero-copy).
  std::vector<std::tuple<void*, void*, size_t>> aliased_reflects;
  for (size_t o = 0; o < num_outputs; ++o) {
    const std::string& name = engine->output_binding_names[o];

    // Aliased output (KV-cache / user): the engine updates the aliased input in
    // place, so bind this output binding to the aliased input's device pointer.
    const int alias_in = engine->output_aliased_input_idx[o];
    if (alias_in >= 0) {
      void* bind_ptr = input_bind_ptrs[static_cast<size_t>(alias_in)];
      if (bind_ptr == nullptr) {
        ET_LOG(Error, "TensorRTBackend::execute: aliased output '%s' has no bound input pointer", name.c_str());
        return Error::InvalidState;
      }
      if (!ctx->setTensorAddress(name.c_str(), bind_ptr)) {
        ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for aliased output '%s'", name.c_str());
        return Error::InvalidState;
      }
      // The aliased output IS a delegate output arg (the caller-owned mutable
      // buffer's mutation slot). Consume it and record a reflect so ExecuTorch's
      // write-back copy_ sees the engine's in-place update.
      const size_t arg_i = arg_idx++;
      EValue* out_arg = args[arg_i];
      TORCHTRT_ET_CHECK_NOT_NULL(
          out_arg, Error::InvalidArgument, "TensorRTBackend::execute: aliased output %zu is not a tensor", o);
      if (!out_arg->isTensor()) {
        ET_LOG(Error, "TensorRTBackend::execute: aliased output %zu is not a tensor", o);
        return Error::InvalidArgument;
      }
      exec_aten::Tensor et_alias_out = out_arg->toTensor();
      nvinfer1::Dims a_dims = ctx->getTensorShape(name.c_str());
      if (a_dims.nbDims >= 0 && a_dims.nbDims <= nvinfer1::Dims::MAX_DIMS) {
        SizesType a_sizes[nvinfer1::Dims::MAX_DIMS];
        for (int d = 0; d < a_dims.nbDims; ++d) {
          a_sizes[d] = static_cast<SizesType>(a_dims.d[d]);
        }
        (void)executorch::runtime::resize_tensor(et_alias_out, {a_sizes, static_cast<size_t>(a_dims.nbDims)});
      }
      void* dst = et_alias_out.nbytes() > 0 ? et_alias_out.mutable_data_ptr() : nullptr;
      if (dst != nullptr && dst != bind_ptr) {
        aliased_reflects.emplace_back(dst, bind_ptr, et_alias_out.nbytes());
      }
      continue;
    }

    const size_t arg_i = arg_idx++; // continue the shared running arg index after the inputs
    EValue* arg = args[arg_i];
    TORCHTRT_ET_CHECK_NOT_NULL(arg, Error::InvalidArgument, "TensorRTBackend::execute: output %zu is not a tensor", o);
    if (!arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: output %zu is not a tensor", o);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_out = arg->toTensor();

    // Update the ExecuTorch tensor shape to the actual TRT output shape.
    // getTensorShape() is valid after inferShapes() has been called.
    nvinfer1::Dims actual_dims = ctx->getTensorShape(name.c_str());
    if (actual_dims.nbDims < 0 || actual_dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
      ET_LOG(Error, "TensorRTBackend::execute: invalid output rank for '%s'", name.c_str());
      return Error::InvalidState;
    }
    SizesType new_sizes[nvinfer1::Dims::MAX_DIMS];
    for (int d = 0; d < actual_dims.nbDims; ++d) {
      new_sizes[d] = static_cast<SizesType>(actual_dims.d[d]);
    }
    Error resize_err = executorch::runtime::resize_tensor(et_out, {new_sizes, static_cast<size_t>(actual_dims.nbDims)});
    if (resize_err != Error::Ok) {
      ET_LOG(Error, "TensorRTBackend::execute: resize_tensor failed for output '%s'", name.c_str());
      return resize_err;
    }

    void* bind_ptr = nullptr;
    if (et_out.nbytes() == 0) {
      if (engine->cached_output_sizes[o] == 0) {
        cuda_err = cudaMalloc(&engine->cached_output_ptrs[o], 1);
        if (cuda_err != cudaSuccess) {
          return Error::MemoryAllocationFailed;
        }
        engine->cached_output_sizes[o] = 1;
      }
      bind_ptr = engine->cached_output_ptrs[o];
    } else if (engine->unified_memory || is_cuda_accessible_ptr(et_out.const_data_ptr())) {
      bind_ptr = et_out.mutable_data_ptr();
    } else {
      const size_t needed = et_out.nbytes();
      if (needed > engine->cached_output_sizes[o]) {
        if (engine->cached_output_ptrs[o] != nullptr) {
          cudaFree(engine->cached_output_ptrs[o]);
        }
        cuda_err = cudaMalloc(&engine->cached_output_ptrs[o], needed);
        if (cuda_err != cudaSuccess) {
          engine->cached_output_ptrs[o] = nullptr;
          engine->cached_output_sizes[o] = 0;
          return Error::MemoryAllocationFailed;
        }
        engine->cached_output_sizes[o] = needed;
      }
      bind_ptr = engine->cached_output_ptrs[o];
      output_staged_to_host = true;
      outputs_needing_copy.push_back({arg_i, bind_ptr});
    }

    if (!ctx->setTensorAddress(name.c_str(), bind_ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for output '%s'", name.c_str());
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 4. Enqueue inference on the current CUDA stream
  // ------------------------------------------------------------------
  if (!ctx->enqueueV3(stream)) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: enqueueV3 failed. If a CUDA green context is "
        "current, scope a CudaStreamGuard with a green-context stream: "
        "cudaStreamPerThread is invalid while a green context is current.");
    return Error::InvalidState;
  }

  // Caller-owned KV: reflect each engine in-place update into its delegate output
  // EValue (D2D on the same stream, after the engine work). No-op list under
  // zero-copy (dst == src filtered out at bind time).
  for (const auto& r : aliased_reflects) {
    cuda_err = cudaMemcpyAsync(std::get<0>(r), std::get<1>(r), std::get<2>(r), cudaMemcpyDeviceToDevice, stream);
    if (cuda_err != cudaSuccess) {
      ET_LOG(
          Error, "TensorRTBackend::execute: aliased-output reflect D2D copy failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
  }

  // The engine work is now in flight on `stream`. Decide whether to wait for it:
  //   must_sync = an output is staged to host (the caller reads the D2H result on
  //   return), an input was staged from host (its async H2D read the caller's host
  //   buffer, which the caller may reuse once we return), or no caller stream is
  //   active (preserve the historical "results ready on return" behavior).
  // Otherwise (caller stream + all I/O device-resident) leave the work enqueued so
  // it composes with the caller's later GPU work, and record inflight_event so the
  // next execute() and the destructor wait before reusing/freeing exec_ctx. The D2H
  // copies live in the must_sync branch: an output staged to host always sets
  // output_staged_to_host, so outputs_needing_copy is empty on the skip path.
  // A non-zero-copy aliased reflect enqueues the engine's in-place update into
  // the delegate output EValue on `stream`; ExecuTorch's buffer-mutation copy_
  // reads that EValue after execute() returns, so the reflect must complete
  // first. Zero-copy aliases (dst == src) record no reflect, so the common
  // caller-owned KV fast path is untouched.
  const bool aliased_reflect_pending = !aliased_reflects.empty();
  const bool must_sync =
      output_staged_to_host || input_staged_from_host || aliased_reflect_pending || !g_user_stream_set;
  if (must_sync) {
    for (auto& output : outputs_needing_copy) {
      exec_aten::Tensor et_out = args[output.first]->toTensor();
      cuda_err =
          cudaMemcpyAsync(et_out.mutable_data_ptr(), output.second, et_out.nbytes(), cudaMemcpyDeviceToHost, stream);
      if (cuda_err != cudaSuccess) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: D2H copy failed for output %zu: %s",
            output.first,
            cudaGetErrorString(cuda_err));
        return Error::InvalidProgram;
      }
    }
    cuda_err = cudaStreamSynchronize(stream);
    engine->inflight_pending = false;
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "TensorRTBackend::execute: cudaStreamSynchronize failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
  } else {
    cuda_err = cudaEventRecord(engine->inflight_event, stream);
    if (cuda_err != cudaSuccess) {
      // Could not arm the completion marker; drain now so a later execute() or the
      // destructor never reconfigures or frees exec_ctx while this enqueue runs.
      ET_LOG(Error, "TensorRTBackend::execute: cudaEventRecord failed: %s", cudaGetErrorString(cuda_err));
      (void)cudaStreamSynchronize(stream);
      engine->inflight_pending = false;
      return Error::InvalidProgram;
    }
    engine->inflight_pending = true;
  }
  return Error::Ok;
}

// ---------------------------------------------------------------------------
// destroy
//
// Explicitly destructs the EngineHandle. The underlying memory was allocated
// by ExecuTorch's MemoryAllocator and will be reclaimed by the arena.
// ---------------------------------------------------------------------------
void TensorRTBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    static_cast<EngineHandle*>(handle)->~EngineHandle();
  }
}

} // namespace executorch_backend
} // namespace torch_tensorrt

// ---------------------------------------------------------------------------
// Static registration – links the name "TensorRTBackend" used in the .pte
// file to this implementation at program startup.
// ---------------------------------------------------------------------------
namespace {

torch_tensorrt::executorch_backend::TensorRTBackend& get_backend() {
  static torch_tensorrt::executorch_backend::TensorRTBackend backend;
  return backend;
}

const ::executorch::runtime::Backend kBackendId{"TensorRTBackend", &get_backend()};
const auto kRegistered = ::executorch::runtime::register_backend(kBackendId);

} // namespace
