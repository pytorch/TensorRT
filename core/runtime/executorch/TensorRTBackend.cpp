/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "core/runtime/executorch/TensorRTBackend.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/log.h>

// RTDevice and Platform must be included before TRTEngine.h because TRTEngine.h
// references them without including their headers directly (Bazel handles this
// via transitive deps, but a standalone compile needs them explicit).
#include "core/runtime/Platform.h"
#include "core/runtime/RTDevice.h"
#include "core/runtime/TRTEngine.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace executorch_backend {

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

namespace {

// ---------------------------------------------------------------------------
// Blob deserialization
//
// Wire format written by
//   py/torch_tensorrt/executorch/serialization.py::serialize_engine_info()
//
//   [uint32_t count (LE)]
//   for each of `count` entries:
//     [uint32_t len (LE)] [uint8_t data[len]]
//
// The resulting vector<string> is passed directly to
//   core::runtime::TRTEngine(std::vector<std::string> serialized_info)
// which expects the 11-element list defined by SerializedInfoIndex in
//   core/runtime/runtime.h
// ---------------------------------------------------------------------------
std::vector<std::string> deserialize_engine_info(const void* data, size_t size) {
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  const uint8_t* const end = ptr + size;

  if (ptr + sizeof(uint32_t) > end) {
    return {};
  }

  uint32_t count = 0;
  std::memcpy(&count, ptr, sizeof(uint32_t));
  ptr += sizeof(uint32_t);

  std::vector<std::string> result;
  result.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    if (ptr + sizeof(uint32_t) > end) {
      return {};
    }
    uint32_t len = 0;
    std::memcpy(&len, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    if (ptr + len > end) {
      return {};
    }
    result.emplace_back(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
  }

  return result;
}

// ---------------------------------------------------------------------------
// Build a nvinfer1::Dims from an ExecuTorch tensor's shape
// ---------------------------------------------------------------------------
nvinfer1::Dims to_trt_dims(const exec_aten::Tensor& t) {
  nvinfer1::Dims dims{};
  dims.nbDims = t.dim();
  for (int d = 0; d < t.dim(); ++d) {
    dims.d[d] = static_cast<int64_t>(t.size(d));
  }
  return dims;
}

} // namespace

// ---------------------------------------------------------------------------
// is_available
// ---------------------------------------------------------------------------
bool TensorRTBackend::is_available() const {
  return true;
}

// ---------------------------------------------------------------------------
// init
//
// Deserializes the processed blob into a TRTEngine and returns it as the
// opaque DelegateHandle.  The engine is placement-new'd into memory
// provided by the ExecuTorch MemoryAllocator so that ExecuTorch owns the
// lifetime; destroy() calls the destructor explicitly.
// ---------------------------------------------------------------------------
Result<DelegateHandle*> TensorRTBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;
  ET_LOG(Info, "TensorRTBackend::init: enter");

  if (!is_available()) {
    ET_LOG(Error, "TensorRT backend is not available");
    return Error::NotSupported;
  }

  if (processed == nullptr || processed->data() == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: null processed buffer");
    return Error::InvalidArgument;
  }

  auto serialized_info = deserialize_engine_info(processed->data(), processed->size());

  if (serialized_info.empty()) {
    fprintf(stderr, "[TensorRTBackend::init] FAIL: deserialize_engine_info returned empty\n");
    ET_LOG(Error, "TensorRTBackend::init: failed to deserialize engine blob");
    return Error::InvalidArgument;
  }
  ET_LOG(Info, "TensorRTBackend::init: deserialized %zu entries", serialized_info.size());

  // Validate the vector length before handing to TRTEngine
  // (verify_serialization_fmt throws on mismatch)
  ET_LOG(Info, "TensorRTBackend::init: calling verify_serialization_fmt");
  try {
    core::runtime::TRTEngine::verify_serialization_fmt(serialized_info);
  } catch (const std::exception& e) {
    ET_LOG(Error, "TensorRTBackend::init: verify_serialization_fmt threw: %s", e.what());
    return Error::InvalidArgument;
  } catch (...) {
    ET_LOG(Error, "TensorRTBackend::init: verify_serialization_fmt threw unknown exception");
    return Error::InvalidArgument;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  if (allocator == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: null runtime allocator");
    return Error::InvalidState;
  }
  ET_LOG(Info, "TensorRTBackend::init: got allocator");

  // Allocate raw storage for TRTEngine from ExecuTorch's arena
  core::runtime::TRTEngine* engine = allocator->allocateInstance<core::runtime::TRTEngine>();
  if (engine == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: allocateInstance failed");
    return Error::MemoryAllocationFailed;
  }
  ET_LOG(Info, "TensorRTBackend::init: allocated engine storage at %p", (void*)engine);

  // Construct in-place; TRTEngine(std::vector<std::string>) deserializes the
  // engine bytes, builds the IRuntime/ICudaEngine/IExecutionContext, and
  // populates in_binding_names / out_binding_names / num_io.
  ET_LOG(Info, "TensorRTBackend::init: constructing TRTEngine in-place");
  try {
    new (engine) core::runtime::TRTEngine(std::move(serialized_info));
  } catch (const std::exception& e) {
    fprintf(stderr, "[TensorRTBackend::init] FAIL: TRTEngine constructor threw: %s\n", e.what());
    ET_LOG(Error, "TensorRTBackend::init: TRTEngine constructor threw: %s", e.what());
    return Error::InvalidArgument;
  } catch (...) {
    fprintf(stderr, "[TensorRTBackend::init] FAIL: TRTEngine constructor threw unknown exception\n");
    ET_LOG(Error, "TensorRTBackend::init: TRTEngine constructor threw unknown exception");
    return Error::InvalidArgument;
  }

  // Release the blob; we no longer need it
  processed->Free();

  ET_LOG(
      Info,
      "TensorRTBackend::init: engine '%s' ready (%zu inputs, %zu outputs)",
      engine->name.c_str(),
      engine->num_io.first,
      engine->num_io.second);

  return static_cast<DelegateHandle*>(engine);
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
  fprintf(stderr, "[TensorRTBackend::execute] enter: handle=%p args.size()=%zu\n", (void*)handle, args.size());
  ET_LOG(Info, "TensorRTBackend::execute: enter");

  if (handle == nullptr) {
    ET_LOG(Error, "TensorRTBackend::execute: null delegate handle");
    return Error::InvalidArgument;
  }
  ET_LOG(Info, "TensorRTBackend::execute: got delegate handle");
  auto* engine = static_cast<core::runtime::TRTEngine*>(handle);

  const size_t num_inputs = engine->num_io.first;
  const size_t num_outputs = engine->num_io.second;
  ET_LOG(Info, "TensorRTBackend::execute: got num_inputs %zu and num_outputs %zu", num_inputs, num_outputs);

  if (args.size() < num_inputs + num_outputs) {
    ET_LOG(
        Error, "TensorRTBackend::execute: expected at least %zu args, got %zu", num_inputs + num_outputs, args.size());
    return Error::InvalidArgument;
  }
  ET_LOG(Info, "TensorRTBackend::execute: got engine");
  // IExecutionContext::enqueueV3 is not thread-safe; use the engine mutex
  std::unique_lock<std::mutex> lock(engine->mu);

  nvinfer1::IExecutionContext* ctx = engine->exec_ctx.get();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(engine->device_info.id));

  // ExecuTorch's portable runtime pre-allocates output tensors as CPU buffers.
  // TRT requires CUDA device pointers for all bindings.  We use
  // cudaPointerGetAttributes to detect CPU pointers and stage them through
  // temporary CUDA allocations, copying back after inference.
  auto is_cuda_ptr = [](const void* ptr) -> bool {
    if (ptr == nullptr)
      return false;
    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    return err == cudaSuccess && attrs.type == cudaMemoryTypeDevice;
  };

  std::vector<void*> temp_input_bufs(num_inputs, nullptr);
  std::vector<void*> temp_output_bufs(num_outputs, nullptr);

  // Cleanup helper – called on every return path.
  auto free_temp = [&]() {
    for (void* p : temp_input_bufs)
      if (p)
        cudaFree(p);
    for (void* p : temp_output_bufs)
      if (p)
        cudaFree(p);
  };

  // ------------------------------------------------------------------
  // 1. Bind input shapes and addresses
  // ------------------------------------------------------------------
  for (size_t i = 0; i < num_inputs; ++i) {
    EValue* arg = args[i];
    if (arg == nullptr || !arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: input %zu is not a tensor", i);
      free_temp();
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_in = arg->toTensor();
    const std::string& name = engine->in_binding_names[i];
    nvinfer1::Dims dims = to_trt_dims(et_in);

    if (!ctx->setInputShape(name.c_str(), dims)) {
      ET_LOG(Error, "TensorRTBackend::execute: setInputShape failed for '%s'", name.c_str());
      free_temp();
      return Error::InvalidState;
    }

    void* src_ptr = et_in.mutable_data_ptr();
    void* trt_ptr = src_ptr;

    static char placeholder[16] = {};
    if (src_ptr == nullptr || et_in.numel() == 0) {
      trt_ptr = placeholder;
    } else if (!is_cuda_ptr(src_ptr)) {
      // CPU input: stage to a temporary CUDA buffer
      size_t nbytes = et_in.nbytes();
      if (cudaMalloc(&temp_input_bufs[i], nbytes) != cudaSuccess) {
        ET_LOG(Error, "TensorRTBackend::execute: cudaMalloc failed for input %zu", i);
        free_temp();
        return Error::MemoryAllocationFailed;
      }
      cudaMemcpyAsync(temp_input_bufs[i], src_ptr, nbytes, cudaMemcpyHostToDevice, stream);
      trt_ptr = temp_input_bufs[i];
    }

    if (!ctx->setTensorAddress(name.c_str(), trt_ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for input '%s'", name.c_str());
      free_temp();
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 2. Infer output shapes (requires all input shapes to be set first)
  // ------------------------------------------------------------------
  {
    const int32_t io_size = engine->cuda_engine->getNbIOTensors();
    std::vector<const char*> unresolved(static_cast<size_t>(io_size), nullptr);
    const int32_t n_unresolved = ctx->inferShapes(io_size, unresolved.data());
    if (n_unresolved != 0) {
      ET_LOG(Error, "TensorRTBackend::execute: inferShapes could not resolve %d tensor(s)", n_unresolved);
      free_temp();
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 3. Bind output addresses
  // ExecuTorch pre-allocates output tensors, but they may be CPU buffers.
  // If so, allocate a temporary CUDA buffer and copy back after inference.
  // ------------------------------------------------------------------
  for (size_t o = 0; o < num_outputs; ++o) {
    EValue* arg = args[num_inputs + o];
    if (arg == nullptr || !arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: output %zu is not a tensor", o);
      free_temp();
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_out = arg->toTensor();
    const std::string& name = engine->out_binding_names[o];
    void* dst_ptr = et_out.mutable_data_ptr();
    void* trt_ptr = dst_ptr;

    if (!is_cuda_ptr(dst_ptr)) {
      // CPU output buffer: allocate temporary CUDA memory for TRT to write into
      size_t nbytes = et_out.nbytes();
      if (cudaMalloc(&temp_output_bufs[o], nbytes) != cudaSuccess) {
        ET_LOG(Error, "TensorRTBackend::execute: cudaMalloc failed for output %zu", o);
        free_temp();
        return Error::MemoryAllocationFailed;
      }
      trt_ptr = temp_output_bufs[o];
    }

    if (!ctx->setTensorAddress(name.c_str(), trt_ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for output '%s'", name.c_str());
      free_temp();
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 4. Enqueue inference on the current CUDA stream
  // ------------------------------------------------------------------
  if (!ctx->enqueueV3(stream)) {
    ET_LOG(Error, "TensorRTBackend::execute: enqueueV3 failed");
    free_temp();
    return Error::InvalidState;
  }

  // ------------------------------------------------------------------
  // 5. Copy temporary CUDA outputs back to the ExecuTorch CPU buffers
  // ------------------------------------------------------------------
  for (size_t o = 0; o < num_outputs; ++o) {
    if (temp_output_bufs[o] != nullptr) {
      exec_aten::Tensor et_out = args[num_inputs + o]->toTensor();
      cudaMemcpyAsync(et_out.mutable_data_ptr(), temp_output_bufs[o], et_out.nbytes(), cudaMemcpyDeviceToHost, stream);
    }
  }

  // Synchronize so outputs are visible to downstream ExecuTorch ops
  cudaStreamSynchronize(stream);

  free_temp();
  return Error::Ok;
}

// ---------------------------------------------------------------------------
// destroy
//
// Explicitly destructs the TRTEngine.  The underlying memory was allocated
// by ExecuTorch's MemoryAllocator and will be reclaimed by the arena.
// ---------------------------------------------------------------------------
void TensorRTBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    static_cast<core::runtime::TRTEngine*>(handle)->~TRTEngine();
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
