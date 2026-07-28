#include "core/plugins/impl/scatter_add_plugin.h"
#include "core/plugins/plugins.h"
#include "core/util/prelude.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

ScatterAddPlugin::ScatterAddPlugin() = default;

nvinfer1::IPluginCapability* ScatterAddPlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
  switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    default:
      return nullptr;
  }
}

nvinfer1::IPluginV3* ScatterAddPlugin::clone() noexcept {
  return new ScatterAddPlugin(*this);
}

// ---------------------------------------------------------------------------
// IPluginV3OneCore
// ---------------------------------------------------------------------------

const char* ScatterAddPlugin::getPluginName() const noexcept {
  return "ScatterAdd";
}

const char* ScatterAddPlugin::getPluginVersion() const noexcept {
  return "1";
}

const char* ScatterAddPlugin::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

// ---------------------------------------------------------------------------
// IPluginV3OneBuild
// ---------------------------------------------------------------------------

int32_t ScatterAddPlugin::getNbOutputs() const noexcept {
  return 1;
}

int32_t ScatterAddPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept {
  // Output has the same dtype as src (input 0).
  outputTypes[0] = inputTypes[0];
  return 0;
}

int32_t ScatterAddPlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    const nvinfer1::DimsExprs* /*shapeInputs*/,
    int32_t /*nbShapeInputs*/,
    nvinfer1::DimsExprs* outputs,
    int32_t /*nbOutputs*/,
    nvinfer1::IExprBuilder& /*exprBuilder*/) noexcept {
  // Output shape == src shape (input 0).
  outputs[0] = inputs[0];
  return 0;
}

bool ScatterAddPlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  const auto& desc = inOut[pos];

  // All tensors must be row-major (linear) layout.
  if (desc.desc.format != nvinfer1::TensorFormat::kLINEAR) {
    return false;
  }

  // Positions 1 through nbInputs-2 are index tensors: int32 or int64.
  if (pos >= 1 && pos <= nbInputs - 2) {
    return desc.desc.type == nvinfer1::DataType::kINT32 || desc.desc.type == nvinfer1::DataType::kINT64;
  }

  // pos 0 (src), pos nbInputs-1 (values), pos nbInputs (output):
  // float32 / float16 / bfloat16, all sharing the same type.
  const bool float_type = desc.desc.type == nvinfer1::DataType::kFLOAT || desc.desc.type == nvinfer1::DataType::kHALF ||
      desc.desc.type == nvinfer1::DataType::kBF16;
  if (!float_type) {
    return false;
  }

  // src, values and output must have the same dtype.
  if (pos == 0) {
    return true;
  }
  return desc.desc.type == inOut[0].desc.type;
}

int32_t ScatterAddPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* /*out*/,
    int32_t /*nbOutputs*/) noexcept {
  dtype_ = in[0].desc.type;
  n_indices_ = nbInputs - 2; // exclude src and values
  idx_dtypes_.resize(n_indices_);
  for (int i = 0; i < n_indices_; ++i) {
    idx_dtypes_[i] = in[1 + i].desc.type;
  }
  return 0;
}

size_t ScatterAddPlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* /*inputs*/,
    int32_t /*nbInputs*/,
    const nvinfer1::DynamicPluginTensorDesc* /*outputs*/,
    int32_t /*nbOutputs*/) const noexcept {
  return 0;
}

// ---------------------------------------------------------------------------
// IPluginV3OneRuntime
// ---------------------------------------------------------------------------

int32_t ScatterAddPlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* /*out*/,
    int32_t /*nbOutputs*/) noexcept {
  dtype_ = in[0].type;
  n_indices_ = nbInputs - 2;
  src_shape_ = util::toVec(in[0].dims);
  val_shape_ = util::toVec(in[nbInputs - 1].dims);
  idx_dtypes_.resize(n_indices_);
  idx_shapes_.resize(n_indices_);
  for (int i = 0; i < n_indices_; ++i) {
    idx_dtypes_[i] = in[1 + i].type;
    idx_shapes_[i] = util::toVec(in[1 + i].dims);
  }
  return 0;
}

int32_t ScatterAddPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* /*workspace*/,
    cudaStream_t stream) noexcept {
  const at::ScalarType float_dtype = util::TRTDataTypeToScalarType(dtype_);
  const auto float_opts = at::TensorOptions().device(at::kCUDA).dtype(float_dtype);

  at::Tensor src = at::from_blob(const_cast<void*>(inputs[0]), src_shape_, float_opts);
  at::Tensor val = at::from_blob(const_cast<void*>(inputs[n_indices_ + 1]), val_shape_, float_opts);

  // Build the indices list — one entry per index tensor, all cast to int64
  // as required by ATen's index_put kernel.
  c10::List<std::optional<at::Tensor>> indices;
  indices.reserve(n_indices_);
  for (int i = 0; i < n_indices_; ++i) {
    const at::ScalarType int_dtype = util::TRTDataTypeToScalarType(idx_dtypes_[i]);
    const auto int_opts = at::TensorOptions().device(at::kCUDA).dtype(int_dtype);
    at::Tensor idx = at::from_blob(const_cast<void*>(inputs[1 + i]), idx_shapes_[i], int_opts);
    indices.push_back(std::optional<at::Tensor>(idx.to(torch::kLong)));
  }

  // Use a separate PyTorch CUDA stream and synchronise with the TRT stream via
  // CUDA events — same pattern as interpolate_plugin.cpp.
  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t start_event;
  cudaEventCreate(&start_event);
  cudaEventRecord(start_event, stream);
  cudaStreamWaitEvent(torch_stream.stream(), start_event, 0);

  // index_put with accumulate=true calls the atomicAdd-based CUDA kernel.
  at::Tensor result = at::index_put(src, indices, val, /*accumulate=*/true);

  at::Tensor out_t = at::from_blob(outputs[0], src_shape_, float_opts);
  out_t.copy_(result);

  cudaEvent_t done_event;
  cudaEventCreate(&done_event);
  cudaEventRecord(done_event, torch_stream.stream());
  cudaStreamWaitEvent(stream, done_event, 0);

  cudaEventDestroy(start_event);
  cudaEventDestroy(done_event);

  return 0;
}

nvinfer1::IPluginV3* ScatterAddPlugin::attachToContext(nvinfer1::IPluginResourceContext* /*context*/) noexcept {
  return clone();
}

const nvinfer1::PluginFieldCollection* ScatterAddPlugin::getFieldsToSerialize() noexcept {
  // No configuration attributes to serialize — shapes and dtype are captured
  // from the tensor descriptors at runtime.
  return &empty_fc_;
}

// ---------------------------------------------------------------------------
// ScatterAddPluginCreator
// ---------------------------------------------------------------------------

ScatterAddPluginCreator::ScatterAddPluginCreator() = default;

const char* ScatterAddPluginCreator::getPluginName() const noexcept {
  return "ScatterAdd";
}

const char* ScatterAddPluginCreator::getPluginVersion() const noexcept {
  return "1";
}

const char* ScatterAddPluginCreator::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

const nvinfer1::PluginFieldCollection* ScatterAddPluginCreator::getFieldNames() noexcept {
  return &fc_;
}

nvinfer1::IPluginV3* ScatterAddPluginCreator::createPlugin(
    const char* /*name*/,
    const nvinfer1::PluginFieldCollection* /*fc*/,
    nvinfer1::TensorRTPhase /*phase*/) noexcept {
  return new ScatterAddPlugin();
}

// Register with the torch_tensorrt namespace.
REGISTER_TORCHTRT_PLUGIN(ScatterAddPluginCreator);

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
