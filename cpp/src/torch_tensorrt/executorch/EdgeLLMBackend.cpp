#include "torch_tensorrt/executorch/EdgeLLMBackend.h"
#include "torch_tensorrt/executorch/EdgeLLMBlobHeader.h"
#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include <cuda_runtime.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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

extern const Error kEdgeLLMRegistrationResult;

Error check_registration() {
  if (kEdgeLLMRegistrationResult != Error::Ok) {
    ET_LOG(
        Error, "EdgeLLMBackend registration failed: %s", ::executorch::runtime::to_string(kEdgeLLMRegistrationResult));
  }
  return kEdgeLLMRegistrationResult;
}

bool is_cuda_accessible(const void* pointer) {
  if (pointer == nullptr) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

std::vector<int64_t> tensor_shape(const ::executorch::aten::Tensor& tensor) {
  std::vector<int64_t> shape;
  shape.reserve(static_cast<std::size_t>(tensor.dim()));
  for (ssize_t dim = 0; dim < tensor.dim(); ++dim) {
    shape.push_back(static_cast<int64_t>(tensor.size(dim)));
  }
  return shape;
}

struct HandleDeleter {
  void operator()(EdgeLLMHandle* handle) const {
    if (handle != nullptr) {
      handle->~EdgeLLMHandle();
    }
  }
};

} // namespace

EdgeLLMHandle::~EdgeLLMHandle() {
  int entry_device = -1;
  const bool restore_device = cudaGetDevice(&entry_device) == cudaSuccess && entry_device != device_id;
  (void)cudaSetDevice(device_id);
  if (inflight_event != nullptr && inflight_pending) {
    (void)cudaEventSynchronize(inflight_event);
    inflight_pending = false;
  }
  vision_runner.reset();
  if (inflight_event != nullptr) {
    (void)cudaEventDestroy(inflight_event);
    inflight_event = nullptr;
  }
  if (restore_device) {
    (void)cudaSetDevice(entry_device);
  }
}

bool EdgeLLMBackend::is_available() const {
  return check_registration() == Error::Ok;
}

Result<DelegateHandle*> EdgeLLMBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;
  if (check_registration() != Error::Ok) {
    return kEdgeLLMRegistrationResult;
  }
  if (processed == nullptr || processed->data() == nullptr) {
    ET_LOG(Error, "EdgeLLMBackend::init: null processed payload");
    return Error::InvalidArgument;
  }

  EdgeLLMBlobHeader edge_header;
  if (!EdgeLLMBlobHeader::parse(processed->data(), processed->size(), edge_header)) {
    ET_LOG(Error, "EdgeLLMBackend::init: invalid EL01 component payload");
    return Error::InvalidProgram;
  }
  const void* nested_blob = EdgeLLMBlobHeader::nested_blob_data(processed->data(), edge_header);
  TensorRTBlobHeader trt_header;
  if (!TensorRTBlobHeader::parse(nested_blob, static_cast<std::size_t>(edge_header.blob_size), trt_header)) {
    ET_LOG(Error, "EdgeLLMBackend::init: invalid nested TensorRT payload");
    return Error::InvalidProgram;
  }
  if (trt_header.input_binding_names.size() != 1 || trt_header.output_binding_names.size() != 1) {
    ET_LOG(Error, "EdgeLLMBackend::init: vision runner requires exactly one input and one output");
    return Error::InvalidProgram;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  if (allocator == nullptr) {
    return Error::InvalidState;
  }
  EdgeLLMHandle* handle = allocator->allocateInstance<EdgeLLMHandle>();
  if (handle == nullptr) {
    return Error::MemoryAllocationFailed;
  }
  new (handle) EdgeLLMHandle();
  std::unique_ptr<EdgeLLMHandle, HandleDeleter> handle_guard(handle);
  handle->device_id = trt_header.device_id;

  if (cudaSetDevice(handle->device_id) != cudaSuccess) {
    ET_LOG(Error, "EdgeLLMBackend::init: failed to select CUDA device %d", handle->device_id);
    return Error::InvalidProgram;
  }
  if (cudaEventCreateWithFlags(&handle->inflight_event, cudaEventDisableTiming | cudaEventBlockingSync) !=
      cudaSuccess) {
    ET_LOG(Error, "EdgeLLMBackend::init: failed to create completion event");
    return Error::InvalidProgram;
  }

  const auto caller_stream = ::executorch::extension::cuda::getCallerStream();
  cudaStream_t stream = caller_stream.value_or(cudaStreamPerThread);
  const void* engine_data = TensorRTBlobHeader::engine_data(nested_blob, trt_header);
  handle->vision_runner = trt_edgellm::executorch::VitExecutorchAdapter::create(
      {engine_data, static_cast<std::size_t>(trt_header.engine_size)}, stream);
  if (!handle->vision_runner) {
    ET_LOG(Error, "EdgeLLMBackend::init: failed to create VitRunner adapter");
    return Error::InvalidProgram;
  }

  processed->Free();
  handle_guard.release();
  return static_cast<DelegateHandle*>(handle);
}

Error EdgeLLMBackend::execute(BackendExecutionContext& context, DelegateHandle* delegate_handle, Span<EValue*> args)
    const {
  (void)context;
  if (delegate_handle == nullptr || args.size() != 2 || args[0] == nullptr || args[1] == nullptr ||
      !args[0]->isTensor() || !args[1]->isTensor()) {
    ET_LOG(Error, "EdgeLLMBackend::execute: expected one tensor input and one tensor output");
    return Error::InvalidArgument;
  }
  auto* handle = static_cast<EdgeLLMHandle*>(delegate_handle);
  std::lock_guard<std::mutex> lock(handle->mu);
  if (handle->inflight_pending) {
    if (cudaEventSynchronize(handle->inflight_event) != cudaSuccess) {
      return Error::InvalidProgram;
    }
    handle->inflight_pending = false;
  }

  auto input = args[0]->toTensor();
  auto output = args[1]->toTensor();
  if (input.scalar_type() != ::executorch::aten::ScalarType::Half ||
      output.scalar_type() != ::executorch::aten::ScalarType::Half || !is_cuda_accessible(input.const_data_ptr()) ||
      !is_cuda_accessible(output.mutable_data_ptr())) {
    ET_LOG(Error, "EdgeLLMBackend::execute: prepared vision input/output must be device-resident FP16 tensors");
    return Error::InvalidArgument;
  }

  const auto caller_stream = ::executorch::extension::cuda::getCallerStream();
  cudaStream_t stream = caller_stream.value_or(cudaStreamPerThread);
  const bool ok = handle->vision_runner->execute(
      {input.mutable_data_ptr(), tensor_shape(input), nvinfer1::DataType::kHALF},
      {output.mutable_data_ptr(), tensor_shape(output), nvinfer1::DataType::kHALF},
      stream);
  if (!ok) {
    return Error::InvalidProgram;
  }

  if (cudaEventRecord(handle->inflight_event, stream) != cudaSuccess) {
    (void)cudaStreamSynchronize(stream);
    return Error::InvalidProgram;
  }
  handle->inflight_pending = true;
  return Error::Ok;
}

void EdgeLLMBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    static_cast<EdgeLLMHandle*>(handle)->~EdgeLLMHandle();
  }
}

} // namespace executorch_backend
} // namespace torch_tensorrt

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

EdgeLLMBackend& get_edge_llm_backend() {
  static EdgeLLMBackend backend;
  return backend;
}

const ::executorch::runtime::Backend kEdgeLLMBackendId{"EdgeLLMBackend", &get_edge_llm_backend()};
const Error kEdgeLLMRegistrationResult = ::executorch::runtime::register_backend(kEdgeLLMBackendId);

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
