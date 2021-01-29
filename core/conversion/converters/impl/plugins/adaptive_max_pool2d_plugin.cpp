#include "adaptive_max_pool2d_plugin.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * AdaptiveMaxPool2dPlugin class implementations
 */

AdaptiveMaxPool2dPlugin::AdaptiveMaxPool2dPlugin(
    std::vector<int64_t> in_shape,
    std::vector<int64_t> out_shape,
    std::vector<int64_t> size,
    std::string mode)
    : in_shape_(in_shape), out_shape_(out_shape), size_(size), mode_(mode) {}

AdaptiveMaxPool2dPlugin::AdaptiveMaxPool2dPlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);

  {
    torch::IValue value;
    input_archive.read("in_shape", value);
    in_shape_ = value.toIntVector();
  }
  {
    torch::IValue value;
    input_archive.read("out_shape", value);
    out_shape_ = value.toIntVector();
  }
  {
    torch::IValue value;
    input_archive.read("size", value);
    size_ = value.toIntVector();
  }
  {
    torch::IValue value;
    input_archive.read("mode", value);
    mode_ = value.toStringRef();
  }
}

std::vector<int64_t> AdaptiveMaxPool2dPlugin::getInputShape() {
  return in_shape_;
}

std::vector<int64_t> AdaptiveMaxPool2dPlugin::getOutputShape() {
  return out_shape_;
}

std::vector<int64_t> AdaptiveMaxPool2dPlugin::getOutputSize() {
  return size_;
}

int AdaptiveMaxPool2dPlugin::getNbOutputs() const {
  return 2;
}

const char* AdaptiveMaxPool2dPlugin::getPluginType() const {
  return "AdaptiveMaxPool2d";
}

const char* AdaptiveMaxPool2dPlugin::getPluginVersion() const {
  return "1";
}

const char* AdaptiveMaxPool2dPlugin::getPluginNamespace() const {
  return "";
}

nvinfer1::IPluginV2DynamicExt* AdaptiveMaxPool2dPlugin::clone() const {
  return new AdaptiveMaxPool2dPlugin(in_shape_, out_shape_, size_, mode_);
}

nvinfer1::DimsExprs AdaptiveMaxPool2dPlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output(inputs[0]);

  for (unsigned int i = 0; i < out_shape_.size(); i++) {
    output.d[i] = exprBuilder.constant(out_shape_[i]);
  }

  return output;
}

nvinfer1::DataType AdaptiveMaxPool2dPlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* inputTypes,
    int nbInputs) const {
  // return DataType::kFLOAT;
  if (index == 0) {
    return nvinfer1::DataType::kFLOAT;
  } else {
    // return nvinfer1::DataType::kFLOAT;
    return nvinfer1::DataType::kINT32;
  }
}

int AdaptiveMaxPool2dPlugin::initialize() {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  tensor_options_ = tensor_options_.device(c10::kCUDA);
  index_tensor_options_ = index_tensor_options_.device(c10::kCUDA);
#else
  tensor_options_ = tensor_options_.device(c10::kCPU);
  index_tensor_options_ = index_tensor_options_.device(c10::kCPU);
#endif

  // c10::kFloat = FLOAT32
  tensor_options_ = tensor_options_.dtype(c10::kFloat);
  index_tensor_options_ = index_tensor_options_.dtype(c10::kLong);

  return 0;
}

void AdaptiveMaxPool2dPlugin::serialize(void* buffer) const {
  std::string data = serializeToString();
  size_t size = getSerializationSize();

  data.copy((char*)buffer, size);
}

std::string AdaptiveMaxPool2dPlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  output_archive.write("in_shape", torch::IValue(in_shape_));
  output_archive.write("out_shape", torch::IValue(out_shape_));
  output_archive.write("size", torch::IValue(size_));
  output_archive.write("mode", torch::IValue(mode_));

  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t AdaptiveMaxPool2dPlugin::getSerializationSize() const {
  return serializeToString().size();
}

bool AdaptiveMaxPool2dPlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) {
  TRTORCH_ASSERT(0 <= pos && pos <= 2, "There should be exactly 3 connections to the plugin - 1 input, 2 output");
  TRTORCH_ASSERT(nbInputs == 1, "Expected a single tensor as input to adaptive_max_pool2d plugin");
  TRTORCH_ASSERT(nbOutputs == 2, "Expected two tensors as output to adaptive_max_pool2d plugin");

  const PluginTensorDesc& in = inOut[0];

  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const PluginTensorDesc& out = inOut[1];

  return (in.type == out.type) && (in.format == out.format);
}

void AdaptiveMaxPool2dPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) {
  dtype_ = DataType::kFLOAT;
}

size_t AdaptiveMaxPool2dPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const {
  return 0;
}

int AdaptiveMaxPool2dPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
  at::Tensor input = at::from_blob((void*)inputs[0], util::toVec(inputDesc->dims), [](void*) {}, tensor_options_);
  at::Tensor output = at::from_blob(
      outputs[0], util::volume(outputDesc->dims), [](void*) {}, tensor_options_);
  at::Tensor indices = at::from_blob(
      outputs[1], util::volume(outputDesc->dims), [](void*) {}, index_tensor_options_);

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  if (mode_ == "adaptive_max_pool2d") {
    at::adaptive_max_pool2d_out(output, indices, input, {size_[0], size_[1]});
  }

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

  at::Tensor output;
  at::Tensor indices;

  if (mode_ == "adaptive_max_pool2d") {
    std::tie(output, indices) = at::adaptive_max_pool2d(input, {size_[0], size_[1]});
  }

  cudaMemcpyAsync(
      outputs[0], output.data_ptr(), util::volume(outputDesc->dims) * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(
      outputs[1], indices.data_ptr(), util::volume(outputDesc->dims) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  free(input_blob);

  return 0;
#endif
}

/*
 * AdaptiveMaxPool2dPluginCreator class implementations
 */
const char* AdaptiveMaxPool2dPluginCreator::getPluginNamespace() const {
  return "";
}

const char* AdaptiveMaxPool2dPluginCreator::getPluginName() const {
  return "AdaptiveMaxPool2d";
}

const char* AdaptiveMaxPool2dPluginCreator::getPluginVersion() const {
  return "1";
}

nvinfer1::IPluginV2* AdaptiveMaxPool2dPluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) {
  return nullptr;
}

AdaptiveMaxPool2dPlugin* AdaptiveMaxPool2dPluginCreator::createPlugin(
    const char* name,
    std::vector<int64_t> in_shape,
    std::vector<int64_t> out_shape,
    std::vector<int64_t> size,
    std::string mode) {
  name_ = name;
  return new AdaptiveMaxPool2dPlugin(in_shape, out_shape, size, mode);
}

nvinfer1::IPluginV2* AdaptiveMaxPool2dPluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) {
  name_ = name;
  return new AdaptiveMaxPool2dPlugin((const char*)serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* AdaptiveMaxPool2dPluginCreator::getFieldNames() {
  return nullptr;
}

REGISTER_TENSORRT_PLUGIN(AdaptiveMaxPool2dPluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch