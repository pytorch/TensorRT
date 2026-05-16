/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt.  Fully self-contained: depends only on the TensorRT C++
 * API, the CUDA runtime, ExecuTorch runtime headers, and the C++ standard
 * library.  No libtorch / ATen / c10 dependencies.
 *
 * Wire format (vector-of-strings, produced by
 *   py/torch_tensorrt/executorch/serialization.py::serialize_engine_info()):
 *
 *   [uint32_t count (LE)]
 *   for each of `count` entries:
 *     [uint32_t len (LE)] [uint8_t data[len]]
 *
 * The slots match the SerializedInfoIndex enum in core/runtime/runtime.h.
 * This backend consumes only the four slots it needs:
 *   DEVICE_IDX(2)               = "id%major%minor%type%name"
 *   ENGINE_IDX(3)               = raw serialized TensorRT engine bytes
 *   INPUT_BINDING_NAMES_IDX(4)  = '%'-separated input names
 *   OUTPUT_BINDING_NAMES_IDX(5) = '%'-separated output names
 */

#include "core/runtime/executorch/TensorRTBackend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

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

// ============================================================================
// Wire-format slot indices (mirror SerializedInfoIndex in
// core/runtime/runtime.h; duplicated here to avoid pulling in libtorch headers)
// ============================================================================
namespace {

constexpr size_t kDeviceIdx = 2;
constexpr size_t kEngineIdx = 3;
constexpr size_t kInputBindingNamesIdx = 4;
constexpr size_t kOutputBindingNamesIdx = 5;
constexpr size_t kMinSerializedSlots = 6;

constexpr char kBindingDelim = '%';
constexpr char kDeviceInfoDelim = '%';

} // namespace

// ============================================================================
// Minimal TensorRT logger forwarding to ExecuTorch logging
// ============================================================================
namespace {

class ETTRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        ET_LOG(Error, "TensorRT: %s", msg);
        break;
      case Severity::kWARNING:
        ET_LOG(Info, "TensorRT [WARN]: %s", msg);
        break;
      default:
        // kINFO and kVERBOSE suppressed to avoid noise
        break;
    }
  }
};

ETTRTLogger& get_trt_logger() {
  static ETTRTLogger logger;
  return logger;
}

} // namespace

// ============================================================================
// Wire-format helpers
// ============================================================================
namespace {

// Decode the vector-of-strings wire format produced by
// py/torch_tensorrt/executorch/serialization.py::serialize_engine_info().
std::vector<std::string> deserialize_engine_info(
    const void* data,
    size_t size) {
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

// Split a delimiter-separated string into non-empty tokens.
std::vector<std::string> split_delim(const std::string& s, char delim) {
  std::vector<std::string> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t end = s.find(delim, start);
    if (end == std::string::npos) {
      if (start < s.size()) {
        out.emplace_back(s.substr(start));
      }
      break;
    }
    if (end > start) {
      out.emplace_back(s.substr(start, end - start));
    }
    start = end + 1;
  }
  return out;
}

// Extract the device id (first '%'-separated token) from the DEVICE_IDX slot.
int parse_device_id(const std::string& device_info, int fallback) {
  if (device_info.empty()) {
    return fallback;
  }
  size_t end = device_info.find(kDeviceInfoDelim);
  std::string id_tok =
      (end == std::string::npos) ? device_info : device_info.substr(0, end);
  try {
    return std::stoi(id_tok);
  } catch (...) {
    return fallback;
  }
}

} // namespace

// ============================================================================
// Tensor / dtype helpers
// ============================================================================
namespace {

// Build nvinfer1::Dims from an ExecuTorch tensor's shape.
nvinfer1::Dims to_trt_dims(const exec_aten::Tensor& t) {
  nvinfer1::Dims dims{};
  if (t.dim() > nvinfer1::Dims::MAX_DIMS) {
    ET_LOG(
        Error,
        "TensorRTBackend: tensor has %d dims, exceeds MAX_DIMS=%d",
        t.dim(),
        nvinfer1::Dims::MAX_DIMS);
    dims.nbDims = 0;
    return dims;
  }
  dims.nbDims = t.dim();
  for (int d = 0; d < t.dim(); ++d) {
    dims.d[d] = static_cast<int64_t>(t.size(d));
  }
  return dims;
}

// Element size for a TRT DataType.
size_t trt_dtype_size(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kBOOL:
      return 1;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kUINT8:
      return 1;
    default:
      ET_LOG(
          Error,
          "TensorRTBackend: unknown dtype %d, assuming 4 bytes",
          static_cast<int>(dtype));
      return 4;
  }
}

// Product of a dims object (returns -1 if any dim is negative/unresolved).
int64_t dims_volume(const nvinfer1::Dims& dims) {
  int64_t vol = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return -1;
    }
    vol *= dims.d[i];
  }
  return vol;
}

} // namespace

// ============================================================================
// TRTDelegateState - fully self-contained, libtorch-free
//
// Owns all TensorRT objects and GPU staging buffers.  Members are declared so
// that C++ destroys exec_ctx before cuda_engine before runtime, which is the
// lifetime order required by TensorRT.
// ============================================================================
namespace {

struct TrtDeleter {
  template <class T>
  void operator()(T* p) const {
    delete p;
  }
};

using UniqueRuntime = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;
using UniqueEngine = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;
using UniqueContext = std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>;

struct TRTDelegateState {
  // --- Engine metadata ---
  int device_id = 0;
  std::vector<std::string> in_binding_names;
  std::vector<std::string> out_binding_names;
  std::mutex mu;

  // --- Grow-only GPU staging buffers (per-binding) ---
  std::vector<void*> input_gpu_bufs;
  std::vector<void*> output_gpu_bufs;
  std::vector<size_t> input_buf_sizes;
  std::vector<size_t> output_buf_sizes;

  // --- TensorRT objects ---
  // Declaration order matters: C++ destroys members in reverse declaration
  // order, so runtime is declared first (destroyed last) and exec_ctx is
  // declared last (destroyed first).  TRT requires
  //   exec_ctx -> cuda_engine -> runtime.
  UniqueRuntime runtime;
  UniqueEngine cuda_engine;
  UniqueContext exec_ctx;
};

} // namespace

// ============================================================================
// is_available
// ============================================================================
bool TensorRTBackend::is_available() const {
  return true;
}

// ============================================================================
// init
// ============================================================================
Result<DelegateHandle*> TensorRTBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;

  if (processed == nullptr || processed->data() == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: null processed buffer");
    return Error::InvalidArgument;
  }

  auto serialized_info =
      deserialize_engine_info(processed->data(), processed->size());

  if (serialized_info.size() < kMinSerializedSlots) {
    ET_LOG(
        Error,
        "TensorRTBackend::init: serialized info has %zu slots, need >= %zu",
        serialized_info.size(),
        kMinSerializedSlots);
    return Error::InvalidArgument;
  }

  const std::string& device_blob = serialized_info[kDeviceIdx];
  const std::string& engine_blob = serialized_info[kEngineIdx];
  const std::string& in_names_blob = serialized_info[kInputBindingNamesIdx];
  const std::string& out_names_blob = serialized_info[kOutputBindingNamesIdx];

  if (engine_blob.empty()) {
    ET_LOG(Error, "TensorRTBackend::init: empty engine bytes");
    return Error::InvalidArgument;
  }

  std::vector<std::string> in_names = split_delim(in_names_blob, kBindingDelim);
  std::vector<std::string> out_names =
      split_delim(out_names_blob, kBindingDelim);

  int device_id = parse_device_id(device_blob, /*fallback=*/0);

  cudaError_t dev_err = cudaSetDevice(device_id);
  if (dev_err != cudaSuccess) {
    ET_LOG(
        Error,
        "TensorRTBackend::init: cudaSetDevice(%d) failed: %s",
        device_id,
        cudaGetErrorString(dev_err));
    return Error::InvalidState;
  }

  UniqueRuntime runtime(nvinfer1::createInferRuntime(get_trt_logger()));
  if (!runtime) {
    ET_LOG(Error, "TensorRTBackend::init: createInferRuntime failed");
    return Error::InvalidState;
  }

  UniqueEngine cuda_engine(runtime->deserializeCudaEngine(
      engine_blob.data(), engine_blob.size()));
  if (!cuda_engine) {
    ET_LOG(Error, "TensorRTBackend::init: deserializeCudaEngine failed");
    return Error::InvalidState;
  }

  // If the wire format didn't carry binding names, query them from the engine.
  if (in_names.empty() && out_names.empty()) {
    const int32_t num_io = cuda_engine->getNbIOTensors();
    for (int32_t i = 0; i < num_io; ++i) {
      const char* tname = cuda_engine->getIOTensorName(i);
      auto mode = cuda_engine->getTensorIOMode(tname);
      if (mode == nvinfer1::TensorIOMode::kINPUT) {
        in_names.emplace_back(tname);
      } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
        out_names.emplace_back(tname);
      }
    }
  }

  // Validate every named binding exists on the engine with the right IO mode.
  for (const auto& n : in_names) {
    if (cuda_engine->getTensorIOMode(n.c_str()) !=
        nvinfer1::TensorIOMode::kINPUT) {
      ET_LOG(
          Error,
          "TensorRTBackend::init: '%s' is not an input tensor in the engine",
          n.c_str());
      return Error::InvalidArgument;
    }
  }
  for (const auto& n : out_names) {
    if (cuda_engine->getTensorIOMode(n.c_str()) !=
        nvinfer1::TensorIOMode::kOUTPUT) {
      ET_LOG(
          Error,
          "TensorRTBackend::init: '%s' is not an output tensor in the engine",
          n.c_str());
      return Error::InvalidArgument;
    }
  }

  UniqueContext exec_ctx(cuda_engine->createExecutionContext());
  if (!exec_ctx) {
    ET_LOG(Error, "TensorRTBackend::init: createExecutionContext failed");
    return Error::InvalidState;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  if (allocator == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: null runtime allocator");
    return Error::InvalidState;
  }

  TRTDelegateState* state = allocator->allocateInstance<TRTDelegateState>();
  if (state == nullptr) {
    ET_LOG(Error, "TensorRTBackend::init: allocateInstance failed");
    return Error::MemoryAllocationFailed;
  }
  new (state) TRTDelegateState{};

  state->device_id = device_id;
  state->in_binding_names = std::move(in_names);
  state->out_binding_names = std::move(out_names);
  state->runtime = std::move(runtime);
  state->cuda_engine = std::move(cuda_engine);
  state->exec_ctx = std::move(exec_ctx);

  // The blob is no longer needed once the engine has been deserialized.
  processed->Free();

  ET_LOG(
      Info,
      "TensorRTBackend::init: ready on device %d (%zu inputs, %zu outputs)",
      state->device_id,
      state->in_binding_names.size(),
      state->out_binding_names.size());

  return static_cast<DelegateHandle*>(state);
}

// ============================================================================
// execute
//
// Uses cudaStreamPerThread (CUDA's per-thread default stream).  That stream
// honors whatever CUDA context the calling thread has current, so callers
// remain free to manage device / context selection externally.
// ============================================================================
Error TensorRTBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  (void)context;

  if (handle == nullptr) {
    ET_LOG(Error, "TensorRTBackend::execute: null delegate handle");
    return Error::InvalidArgument;
  }
  auto* state = static_cast<TRTDelegateState*>(handle);

  cudaError_t dev_err = cudaSetDevice(state->device_id);
  if (dev_err != cudaSuccess) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: cudaSetDevice(%d) failed: %s",
        state->device_id,
        cudaGetErrorString(dev_err));
    return Error::InvalidState;
  }

  const size_t num_inputs = state->in_binding_names.size();
  const size_t num_outputs = state->out_binding_names.size();

  if (args.size() < num_inputs + num_outputs) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: expected at least %zu args, got %zu",
        num_inputs + num_outputs,
        args.size());
    return Error::InvalidArgument;
  }

  // IExecutionContext is not thread-safe; serialize concurrent calls.
  std::unique_lock<std::mutex> lock(state->mu);

  cudaStream_t stream = cudaStreamPerThread;
  nvinfer1::IExecutionContext* ctx = state->exec_ctx.get();

  // --------------------------------------------------------------------
  // 1. Bind input shapes and addresses (stage CPU buffers to GPU as needed)
  // --------------------------------------------------------------------
  for (size_t i = 0; i < num_inputs; ++i) {
    EValue* arg = args[i];
    if (arg == nullptr || !arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: input %zu is not a tensor", i);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_in = arg->toTensor();
    const std::string& name = state->in_binding_names[i];
    nvinfer1::Dims dims = to_trt_dims(et_in);

    if (!ctx->setInputShape(name.c_str(), dims)) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: setInputShape failed for '%s'",
          name.c_str());
      return Error::InvalidState;
    }

    void* host_ptr = et_in.mutable_data_ptr();
    size_t nbytes = et_in.nbytes();

    // TensorRT requires a non-null address even for 0-element tensors.
    // Thread-local placeholder avoids sharing across concurrent delegates.
    static thread_local char placeholder[16] = {};
    if (host_ptr == nullptr || et_in.numel() == 0) {
      host_ptr = placeholder;
      nbytes = 0;
    }

    void* gpu_ptr = nullptr;
    cudaPointerAttributes attrs;
    cudaError_t attr_err = cudaPointerGetAttributes(&attrs, host_ptr);
    if (attr_err == cudaSuccess &&
        (attrs.type == cudaMemoryTypeDevice ||
         attrs.type == cudaMemoryTypeManaged)) {
      gpu_ptr = host_ptr;
    } else {
      // Clear any spurious error from probing a host pointer.
      cudaGetLastError();

      if (i >= state->input_gpu_bufs.size()) {
        state->input_gpu_bufs.resize(i + 1, nullptr);
        state->input_buf_sizes.resize(i + 1, 0);
      }

      if (nbytes == 0) {
        gpu_ptr = placeholder;
      } else {
        // Grow-only reallocation of the per-binding staging buffer.
        if (state->input_buf_sizes[i] < nbytes) {
          if (state->input_gpu_bufs[i]) {
            cudaFree(state->input_gpu_bufs[i]);
            state->input_gpu_bufs[i] = nullptr;
          }
          cudaError_t alloc_err =
              cudaMalloc(&state->input_gpu_bufs[i], nbytes);
          if (alloc_err != cudaSuccess) {
            ET_LOG(
                Error,
                "TensorRTBackend::execute: cudaMalloc(%zu) failed for input "
                "'%s': %s",
                nbytes,
                name.c_str(),
                cudaGetErrorString(alloc_err));
            return Error::MemoryAllocationFailed;
          }
          state->input_buf_sizes[i] = nbytes;
        }
        gpu_ptr = state->input_gpu_bufs[i];
        cudaError_t cpy_err = cudaMemcpyAsync(
            gpu_ptr, host_ptr, nbytes, cudaMemcpyHostToDevice, stream);
        if (cpy_err != cudaSuccess) {
          ET_LOG(
              Error,
              "TensorRTBackend::execute: H2D copy failed for input '%s': %s",
              name.c_str(),
              cudaGetErrorString(cpy_err));
          return Error::InvalidState;
        }
      }
    }

    if (!ctx->setTensorAddress(name.c_str(), gpu_ptr)) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: setTensorAddress failed for input '%s'",
          name.c_str());
      return Error::InvalidState;
    }
  }

  // --------------------------------------------------------------------
  // 2. Infer output shapes (all input shapes must be set first)
  // --------------------------------------------------------------------
  {
    const int32_t io_size = state->cuda_engine->getNbIOTensors();
    std::vector<const char*> unresolved(static_cast<size_t>(io_size), nullptr);
    const int32_t n_unresolved =
        ctx->inferShapes(io_size, unresolved.data());
    if (n_unresolved != 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: inferShapes could not resolve %d "
          "tensor(s)",
          n_unresolved);
      return Error::InvalidState;
    }
  }

  // --------------------------------------------------------------------
  // 3. Bind output addresses (size GPU buffers from TRT's inferred shape,
  //    not from the pre-allocated ExecuTorch tensor, to avoid overrun on
  //    dynamic-shape outputs)
  // --------------------------------------------------------------------
  struct OutputCopyInfo {
    void* gpu_ptr = nullptr;
    size_t actual_bytes = 0;
    bool needs_copy = false;
  };
  std::vector<OutputCopyInfo> output_copies(num_outputs);

  for (size_t o = 0; o < num_outputs; ++o) {
    EValue* arg = args[num_inputs + o];
    if (arg == nullptr || !arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: output %zu is not a tensor", o);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_out = arg->toTensor();
    const std::string& name = state->out_binding_names[o];
    void* host_ptr = et_out.mutable_data_ptr();

    nvinfer1::Dims actual_dims = ctx->getTensorShape(name.c_str());
    nvinfer1::DataType out_dtype =
        state->cuda_engine->getTensorDataType(name.c_str());
    int64_t vol = dims_volume(actual_dims);
    if (vol < 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: unresolved output shape for '%s'",
          name.c_str());
      return Error::InvalidState;
    }
    size_t actual_bytes =
        static_cast<size_t>(vol) * trt_dtype_size(out_dtype);
    output_copies[o].actual_bytes = actual_bytes;

    void* gpu_ptr = nullptr;
    if (host_ptr != nullptr && et_out.numel() != 0) {
      cudaPointerAttributes attrs;
      cudaError_t attr_err = cudaPointerGetAttributes(&attrs, host_ptr);
      if (attr_err == cudaSuccess &&
          (attrs.type == cudaMemoryTypeDevice ||
           attrs.type == cudaMemoryTypeManaged)) {
        gpu_ptr = host_ptr;
      } else {
        cudaGetLastError();
      }
    }

    if (gpu_ptr == nullptr) {
      if (o >= state->output_gpu_bufs.size()) {
        state->output_gpu_bufs.resize(o + 1, nullptr);
        state->output_buf_sizes.resize(o + 1, 0);
      }
      // TRT requires a valid 16-byte-aligned device address even for
      // zero-element bindings; round up to 16 bytes.
      size_t required = actual_bytes > 0 ? actual_bytes : 16;
      if (state->output_buf_sizes[o] < required) {
        if (state->output_gpu_bufs[o]) {
          cudaFree(state->output_gpu_bufs[o]);
          state->output_gpu_bufs[o] = nullptr;
        }
        cudaError_t alloc_err =
            cudaMalloc(&state->output_gpu_bufs[o], required);
        if (alloc_err != cudaSuccess) {
          ET_LOG(
              Error,
              "TensorRTBackend::execute: cudaMalloc(%zu) failed for output "
              "'%s': %s",
              required,
              name.c_str(),
              cudaGetErrorString(alloc_err));
          return Error::MemoryAllocationFailed;
        }
        state->output_buf_sizes[o] = required;
      }
      gpu_ptr = state->output_gpu_bufs[o];
      output_copies[o].needs_copy = (host_ptr != nullptr && actual_bytes > 0);
    }

    output_copies[o].gpu_ptr = gpu_ptr;

    if (!ctx->setTensorAddress(name.c_str(), gpu_ptr)) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: setTensorAddress failed for output '%s'",
          name.c_str());
      return Error::InvalidState;
    }
  }

  // --------------------------------------------------------------------
  // 4. Enqueue inference
  // --------------------------------------------------------------------
  if (!ctx->enqueueV3(stream)) {
    ET_LOG(Error, "TensorRTBackend::execute: enqueueV3 failed");
    return Error::InvalidState;
  }

  // --------------------------------------------------------------------
  // 5. Resize output tensors to TRT's inferred shape, then queue any
  //    device-to-host copies asynchronously so multiple small outputs
  //    don't pay one host sync each.
  // --------------------------------------------------------------------
  bool any_d2h_copy = false;
  for (size_t o = 0; o < num_outputs; ++o) {
    const std::string& out_name = state->out_binding_names[o];
    nvinfer1::Dims actual_dims = ctx->getTensorShape(out_name.c_str());
    exec_aten::Tensor et_out = args[num_inputs + o]->toTensor();

    if (et_out.dim() != actual_dims.nbDims) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: rank mismatch for output '%s': "
          "tensor has %d dims, TRT produced %d dims",
          out_name.c_str(),
          et_out.dim(),
          actual_dims.nbDims);
      return Error::InvalidState;
    }

    bool shape_changed = false;
    for (int d = 0; d < actual_dims.nbDims; ++d) {
      if (et_out.size(d) != actual_dims.d[d]) {
        shape_changed = true;
        break;
      }
    }
    if (shape_changed) {
      std::vector<exec_aten::SizesType> new_sizes(actual_dims.nbDims);
      for (int d = 0; d < actual_dims.nbDims; ++d) {
        new_sizes[d] = static_cast<exec_aten::SizesType>(actual_dims.d[d]);
      }
      Error resize_err = ::executorch::runtime::resize_tensor(
          et_out, {new_sizes.data(), new_sizes.size()});
      if (resize_err != Error::Ok) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: resize_tensor failed for output '%s'",
            out_name.c_str());
        return resize_err;
      }
    }

    if (output_copies[o].needs_copy) {
      // Re-fetch pointer/size after potential resize.
      void* host_ptr = et_out.mutable_data_ptr();
      size_t host_bytes = et_out.nbytes();
      size_t copy_bytes =
          std::min(output_copies[o].actual_bytes, host_bytes);
      if (copy_bytes > 0 && host_ptr != nullptr) {
        cudaError_t cpy_err = cudaMemcpyAsync(
            host_ptr,
            output_copies[o].gpu_ptr,
            copy_bytes,
            cudaMemcpyDeviceToHost,
            stream);
        if (cpy_err != cudaSuccess) {
          ET_LOG(
              Error,
              "TensorRTBackend::execute: D2H copy failed for output '%s': %s",
              out_name.c_str(),
              cudaGetErrorString(cpy_err));
          return Error::InvalidState;
        }
        any_d2h_copy = true;
      }
    }
  }

  // Single trailing sync: covers enqueueV3 + all D2H copies.  If there were
  // no D2H copies but the outputs live on the GPU, callers must sync the
  // per-thread stream themselves before reading on the host - matches normal
  // CUDA semantics.
  if (any_d2h_copy) {
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: cudaStreamSynchronize failed: %s",
          cudaGetErrorString(sync_err));
      return Error::InvalidState;
    }
  }

  return Error::Ok;
}

// ============================================================================
// destroy
// ============================================================================
void TensorRTBackend::destroy(DelegateHandle* handle) const {
  if (handle == nullptr) {
    return;
  }
  auto* state = static_cast<TRTDelegateState*>(handle);

  // Make sure all in-flight GPU work on this thread's stream is done before
  // freeing buffers / destroying TRT objects.
  cudaStreamSynchronize(cudaStreamPerThread);

  for (void* buf : state->input_gpu_bufs) {
    if (buf) {
      cudaFree(buf);
    }
  }
  for (void* buf : state->output_gpu_bufs) {
    if (buf) {
      cudaFree(buf);
    }
  }

  // Member declaration order ensures exec_ctx -> cuda_engine -> runtime.
  state->~TRTDelegateState();
  // The underlying arena reclaims the storage.
}

} // namespace executorch_backend
} // namespace torch_tensorrt

// ============================================================================
// Static registration - binds the name "TensorRTBackend" used in the .pte
// file to this implementation at program startup.
// ============================================================================
namespace {

torch_tensorrt::executorch_backend::TensorRTBackend& get_backend() {
  static torch_tensorrt::executorch_backend::TensorRTBackend backend;
  return backend;
}

const ::executorch::runtime::Backend kBackendId{
    "TensorRTBackend",
    &get_backend()};
const auto kRegistered = ::executorch::runtime::register_backend(kBackendId);

} // namespace
