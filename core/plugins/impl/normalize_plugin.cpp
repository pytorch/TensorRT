#include "core/plugins/impl/normalize_plugin.h"
#include "NvInferPlugin.h"
#include "NvInferPluginUtils.h"
#include "core/plugins/plugins.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

/*
 * NormalizePlugin class implementations
 */

NormalizePlugin::NormalizePlugin(int32_t order, std::vector<int32_t> axes, int32_t keep_dims)
    : order_(order), axes_(axes), keep_dims_(keep_dims) {}

NormalizePlugin::NormalizePlugin(const char* data, size_t length) {
  std::istringstream data_stream(std::string(data, length));

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(data_stream);
  {
    torch::IValue value;
    input_archive.read("order", value);
    order_ = (int32_t)value.toInt();
  }
  {
    torch::IValue value;
    input_archive.read("axes", value);
    auto values = value.toIntVector();
    std::vector<int32_t> doubleVec(values.begin(), values.end());
    // axes_ = doubleVec;
    axes_.assign(doubleVec.begin(), doubleVec.end());
  }
  {
    torch::IValue value;
    input_archive.read("keep_dims", value);
    keep_dims_ = (int32_t)value.toInt();
  }
}

int NormalizePlugin::getNbOutputs() const noexcept {
  return 1;
}

const char* NormalizePlugin::getPluginType() const noexcept {
  return "NormalizePlugin";
}

const char* NormalizePlugin::getPluginVersion() const noexcept {
  return "1";
}

const char* NormalizePlugin::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

nvinfer1::IPluginV2DynamicExt* NormalizePlugin::clone() const noexcept {
  return new NormalizePlugin(order_, axes_, keep_dims_);
}

nvinfer1::DimsExprs NormalizePlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  nvinfer1::DimsExprs output;
  output.nbDims = keep_dims_ ? inputs[0].nbDims : inputs[0].nbDims - axes_.size();

  // For order-0 norm, when the norm dimension is None, it should normalize across all dimensions.
  // TODO: For dim=None, the axes_ passed would have [0, 0, 0] which is obtained through loop counter in Torch-TensorRT.
  // Resolve this. For dim=None case, change the axes_ inplace to range(0, axes_.size())
  bool isAxisNone = std::all_of(axes_.begin(), axes_.end(), [](int32_t i) { return i == 0; }) &&
      ((int32_t)axes_.size() == inputs[0].nbDims);
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
    const noexcept {
  return nvinfer1::DataType::kFLOAT;
}

int NormalizePlugin::initialize() noexcept {
  return 0;
}

void NormalizePlugin::serialize(void* buffer) const noexcept {
  std::string data = serializeToString();
  size_t size = getSerializationSize();
  data.copy((char*)buffer, size);
}

std::string NormalizePlugin::serializeToString() const noexcept {
  torch::serialize::OutputArchive output_archive;
  std::vector<int64_t> axesVec(axes_.begin(), axes_.end());
  output_archive.write("order", torch::IValue((int64_t)order_));
  output_archive.write("axes", torch::IValue(axesVec));
  output_archive.write("keep_dims", torch::IValue((int64_t)keep_dims_));
  std::ostringstream data_str;
  output_archive.save_to(data_str);

  return data_str.str();
}

size_t NormalizePlugin::getSerializationSize() const noexcept {
  return serializeToString().size();
}

bool NormalizePlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) noexcept {
  if (pos < 0 || pos > 1) {
    LOG_ERROR("There should be exactly 2 connections to the plugin - 1 input, 1 output");
  }
  if (nbInputs != 1) {
    LOG_ERROR("Expected a single tensor as input to normalize plugin");
  }
  if (nbOutputs != 1) {
    LOG_ERROR("Expected a single tensor as output to normalize plugin");
  }

  const nvinfer1::PluginTensorDesc& in = inOut[0];

  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const nvinfer1::PluginTensorDesc& out = inOut[1];

  return (in.type == out.type) && (in.format == out.format);
}

void NormalizePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int nbOutputs) noexcept {
  dtype_ = nvinfer1::DataType::kFLOAT;
}

size_t NormalizePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const noexcept {
  return 0;
}

int NormalizePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  at::Tensor input =
      at::from_blob((void*)inputs[0], util::toVec(inputDesc->dims), [](void*) {}, {at::kCUDA}).to(torch::kFloat);
  at::Tensor output =
      at::from_blob(outputs[0], util::toVec(outputDesc->dims), [](void*) {}, {at::kCUDA}).to(torch::kFloat);

  at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
  at::cuda::CUDAStreamGuard torch_guard(torch_stream);

  cudaEvent_t event;
  cudaEventCreate(&event);
  cudaEventRecord(event, stream);

  cudaStreamWaitEvent(torch_stream.stream(), event, 0);

  std::vector<int64_t> axes_double(axes_.begin(), axes_.end());
  at::Tensor result = at::norm(input, (int64_t)order_, axes_double, (bool)keep_dims_);
  output.copy_(result);
  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);
  return 0;
}

/*
 * NormalizePluginCreator class implementations
 */
NormalizePluginCreator::NormalizePluginCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("order", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("axes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("keep_dims", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* NormalizePluginCreator::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

const char* NormalizePluginCreator::getPluginName() const noexcept {
  return "NormalizePlugin";
}

const char* NormalizePluginCreator::getPluginVersion() const noexcept {
  return "1";
}

nvinfer1::IPluginV2* NormalizePluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc) noexcept {
  int32_t order = 0;
  std::vector<int32_t> axes;
  int32_t keep_dims = 0;
  for (int i = 0; i < fc->nbFields; i++) {
    std::string field_name(fc->fields[i].name);
    if (field_name.compare("order") == 0) {
      order = *static_cast<const int32_t*>(fc->fields[i].data);
    } else if (field_name.compare("axes") == 0) {
      auto axes_values = static_cast<const int32_t*>(fc->fields[i].data);
      axes.assign(axes_values, axes_values + fc->fields[i].length);
    } else if (field_name.compare("keep_dims") == 0) {
      keep_dims = *static_cast<const int32_t*>(fc->fields[i].data);
    }
  }
  NormalizePlugin* plugin = new NormalizePlugin(order, axes, keep_dims);
  return plugin;
}

nvinfer1::IPluginV2* NormalizePluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
  name_ = name;
  auto plugin = new NormalizePlugin((const char*)serialData, serialLength);
  return plugin;
}

const nvinfer1::PluginFieldCollection* NormalizePluginCreator::getFieldNames() noexcept {
  return nullptr;
}

REGISTER_TORCHTRT_PLUGIN(NormalizePluginCreator);

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
