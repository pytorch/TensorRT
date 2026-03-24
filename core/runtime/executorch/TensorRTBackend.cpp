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
Result<DelegateHandle*> TensorRTBackend::init(BackendInitContext& context, FreeableBuffer* processed) const {
  if (processed == nullptr || processed->data() == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: null processed buffer");
    return Error::InvalidArgument;
  }

  auto serialized_info = deserialize_engine_info(processed->data(), processed->size());

  if (serialized_info.empty()) {
    ET_LOG(Error, "TensorRTBackend::init: failed to deserialize engine blob");
    return Error::InvalidArgument;
  }

  // Validate the vector length before handing to TRTEngine
  // (verify_serialization_fmt throws on mismatch)
  core::runtime::TRTEngine::verify_serialization_fmt(serialized_info);

  MemoryAllocator* allocator = context.get_runtime_allocator();
  if (allocator == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: null runtime allocator");
    return Error::InvalidState;
  }

  // Allocate raw storage for TRTEngine from ExecuTorch's arena
  core::runtime::TRTEngine* engine = allocator->allocateInstance<core::runtime::TRTEngine>();
  if (engine == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: allocateInstance failed");
    return Error::MemoryAllocationFailed;
  }

  // Construct in-place; TRTEngine(std::vector<std::string>) deserializes the
  // engine bytes, builds the IRuntime/ICudaEngine/IExecutionContext, and
  // populates in_binding_names / out_binding_names / num_io.
  new (engine) core::runtime::TRTEngine(std::move(serialized_info));

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

  if (handle == nullptr) {
    ET_LOG(Error, "TensorRTBackend::execute: null delegate handle");
    return Error::InvalidArgument;
  }

  auto* engine = static_cast<core::runtime::TRTEngine*>(handle);

  const size_t num_inputs = engine->num_io.first;
  const size_t num_outputs = engine->num_io.second;

  if (args.size() < num_inputs + num_outputs) {
    ET_LOG(
        Error, "TensorRTBackend::execute: expected at least %zu args, got %zu", num_inputs + num_outputs, args.size());
    return Error::InvalidArgument;
  }

  // IExecutionContext::enqueueV3 is not thread-safe; use the engine mutex
  std::unique_lock<std::mutex> lock(engine->mu);

  nvinfer1::IExecutionContext* ctx = engine->exec_ctx.get();

  // ------------------------------------------------------------------
  // 1. Bind input shapes and addresses
  // ------------------------------------------------------------------
  for (size_t i = 0; i < num_inputs; ++i) {
    EValue* arg = args[i];
    if (arg == nullptr || !arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: input %zu is not a tensor", i);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_in = arg->toTensor();
    const std::string& name = engine->in_binding_names[i];
    nvinfer1::Dims dims = to_trt_dims(et_in);

    if (!ctx->setInputShape(name.c_str(), dims)) {
      ET_LOG(Error, "TensorRTBackend::execute: setInputShape failed for '%s'", name.c_str());
      return Error::InvalidState;
    }

    void* ptr = et_in.mutable_data_ptr();
    // TRT requires a non-null address even for 0-element tensors
    static char placeholder[16] = {};
    if (ptr == nullptr || et_in.numel() == 0) {
      ptr = placeholder;
    }

    if (!ctx->setTensorAddress(name.c_str(), ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for input '%s'", name.c_str());
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
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 3. Bind output addresses (ExecuTorch pre-allocates the buffers)
  // ------------------------------------------------------------------
  for (size_t o = 0; o < num_outputs; ++o) {
    EValue* arg = args[num_inputs + o];
    if (arg == nullptr || !arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: output %zu is not a tensor", o);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_out = arg->toTensor();
    const std::string& name = engine->out_binding_names[o];
    void* ptr = et_out.mutable_data_ptr();

    if (!ctx->setTensorAddress(name.c_str(), ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for output '%s'", name.c_str());
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 4. Enqueue inference on the current CUDA stream
  // ------------------------------------------------------------------
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(static_cast<c10::DeviceIndex>(engine->device_info.id));

  if (!ctx->enqueueV3(stream)) {
    ET_LOG(Error, "TensorRTBackend::execute: enqueueV3 failed");
    return Error::InvalidState;
  }

  // Synchronize so that outputs are visible to downstream ExecuTorch ops
  cudaStreamSynchronize(stream);

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
