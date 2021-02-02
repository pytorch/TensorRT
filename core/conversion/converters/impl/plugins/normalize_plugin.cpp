#include "normalize_plugin.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * NormalizePlugin class implementations
 */

NormalizePlugin::NormalizePlugin(int64_t order, std::vector<int64_t> axes, bool keep_dims)
    : order_(order), axes_(axes), keep_dims_(keep_dims) {}

NormalizePlugin::NormalizePlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
  {
    torch::IValue value;
    input_archive.read("order", value);
    order_ = value.toInt();
  }
  {
    torch::IValue value;
    input_archive.read("axes", value);
    axes_ = value.toIntVector();
  }
  {
    torch::IValue value;
    input_archive.read("keep_dims", value);
    keep_dims_ = value.toBool();
  }
}

int NormalizePlugin::getNbOutputs() const {
  return 1;
}

const char* NormalizePlugin::getPluginType() const {
  return "Normalize";
}

const char* NormalizePlugin::getPluginVersion() const {
  return "1";
}

const char* NormalizePlugin::getPluginNamespace() const {
  return "";
}

nvinfer1::IPluginV2DynamicExt* NormalizePlugin::clone() const {
  return new NormalizePlugin(order_, axes_, keep_dims_);
}

nvinfer1::DimsExprs NormalizePlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output;
  output.nbDims = keep_dims_ ? inputs[0].nbDims : inputs[0].nbDims - axes_.size();

  // For order-0 norm, when the norm dimension is None, it should normalize across all dimensions.
  // TODO: For dim=None, the axes_ passed would have [0, 0, 0] which is obtained through loop counter in TRTorch.
  // Resolve this. For dim=None case, change the axes_ inplace to range(0, axes_.size())
  bool isAxisNone =
      std::all_of(axes_.begin(), axes_.end(), [](int64_t i) { return i == 0; }) && (axes_.size() == inputs[0].nbDims);
  if (isAxisNone) {
    std::iota(axes_.data(), axes_.data() + axes_.size(), 0);
  }
  int64_t out_idx = 0;
  for (int64_t i = 0; i < inputs[0].nbDims; i++) {
    if (std::find(axes_.begin(), axes_.end(), i) != axes_.end()) {
      if (keep_dims_) {
        output.d[out_idx] = exprBuilder.constant(1);
        out_idx += 1;
      }
    } else {
      if (!isAxisNone) {
        output.d[out_idx] = exprBuilder.constant(inputs[0].d[i]->getConstantValue());
      } else {
        output.d[out_idx] = exprBuilder.constant(1);
      }
      out_idx += 1;
    }
  }

  return output;
}

nvinfer1::DataType NormalizePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
    const {
  return DataType::kFLOAT;
}

int NormalizePlugin::initialize() {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  tensor_options_ = tensor_options_.device(c10::kCUDA);
#else
  tensor_options_ = tensor_options_.device(c10::kCPU);
#endif

  // c10::kFloat = FLOAT32
  tensor_options_ = tensor_options_.dtype(c10::kFloat);

  return 0;
}

void NormalizePlugin::serialize(void* buffer) const {
  std::string data = serializeToString();
  size_t size = getSerializationSize();
  data.copy((char*)buffer, size);
}

std::string NormalizePlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  output_archive.write("order", torch::IValue(order_));
  output_archive.write("axes", torch::IValue(axes_));
  output_archive.write("keep_dims", torch::IValue(keep_dims_));
  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t NormalizePlugin::getSerializationSize() const {
  return serializeToString().size();
}

bool NormalizePlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) {
  TRTORCH_ASSERT(0 <= pos && pos <= 1, "There should be exactly 2 connections to the plugin - 1 input, 1 output");
  TRTORCH_ASSERT(nbInputs == 1, "Expected a single tensor as input to normalize plugin");
  TRTORCH_ASSERT(nbOutputs == 1, "Expected a single tensor as output to normalize plugin");

  const PluginTensorDesc& in = inOut[0];

  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const PluginTensorDesc& out = inOut[1];

  return (in.type == out.type) && (in.format == out.format);
}

void NormalizePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) {
  dtype_ = DataType::kFLOAT;
}

size_t NormalizePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const {
  return 0;
}

int NormalizePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
  // TRT <= 7.0
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  at::Tensor input = at::from_blob((void*)inputs[0], util::toVec(inputDesc->dims), [](void*) {}, tensor_options_);
  at::Tensor output = at::from_blob(outputs[0], util::volume(outputDesc->dims), [](void*) {}, tensor_options_);

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);
  at::Tensor result = at::norm(input, order_, axes, keep_dims_);
  output.copy_(result);
  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);
  return 0;
#else
  // TODO: When PyTorch updates to cuDNN 8 try moving back to CUDA based ATen
  // kernels HACK: WAR because there is a segfault if you try to create a CUDA
  // Tensor in the context of TensorRT execution
  float* input_blob = (float*)malloc(util::volume(inputDesc->dims) * sizeof(float));
  cudaMemcpyAsync(
      input_blob,
      static_cast<const void*>(inputs[0]),
      util::volume(inputDesc->dims) * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream);
  cudaStreamSynchronize(stream);

  at::Tensor input = at::from_blob((void*)input_blob, util::toVec(inputDesc->dims), tensor_options_);
  at::Tensor output = at::norm(input, order_, axes_, keep_dims_);

  cudaMemcpyAsync(
      outputs[0], output.data_ptr(), util::volume(outputDesc->dims) * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  free(input_blob);
  return 0;
#endif
}

/*
 * NormalizePluginCreator class implementations
 */
const char* NormalizePluginCreator::getPluginNamespace() const {
  return "";
}

const char* NormalizePluginCreator::getPluginName() const {
  return "Normalize";
}

const char* NormalizePluginCreator::getPluginVersion() const {
  return "1";
}

nvinfer1::IPluginV2* NormalizePluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) {
  return nullptr;
}

NormalizePlugin* NormalizePluginCreator::createPlugin(
    const char* name,
    int64_t order,
    std::vector<int64_t> axes,
    bool keep_dims) {
  name_ = name;
  return new NormalizePlugin(order, axes, keep_dims);
}

nvinfer1::IPluginV2* NormalizePluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) {
  name_ = name;
  return new NormalizePlugin((const char*)serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* NormalizePluginCreator::getFieldNames() {
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(NormalizePluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
