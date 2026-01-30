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
    : order_(order), axes_(axes), keep_dims_(keep_dims) {
  dtype_ = nvinfer1::DataType::kFLOAT;
  setupSerialization();
}

NormalizePlugin::NormalizePlugin(const nvinfer1::PluginFieldCollection* fc) {
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
  
  order_ = order;
  axes_ = axes;
  keep_dims_ = keep_dims;
  dtype_ = nvinfer1::DataType::kFLOAT;
  setupSerialization();
}

void NormalizePlugin::setupSerialization() {
  mDataToSerialize.clear();
  mDataToSerialize.emplace_back("order", &order_, nvinfer1::PluginFieldType::kINT32, 1);
  mDataToSerialize.emplace_back("axes", axes_.data(), nvinfer1::PluginFieldType::kINT32, axes_.size());
  mDataToSerialize.emplace_back("keep_dims", &keep_dims_, nvinfer1::PluginFieldType::kINT32, 1);
  mFCToSerialize.nbFields = mDataToSerialize.size();
  mFCToSerialize.fields = mDataToSerialize.data();
}

// IPluginV3 methods
nvinfer1::IPluginCapability* NormalizePlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
  switch (type) {
    case nvinfer1::PluginCapabilityType::kCORE:
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
  }
  return nullptr;
}

nvinfer1::IPluginV3* NormalizePlugin::clone() noexcept {
  auto plugin = new NormalizePlugin(order_, axes_, keep_dims_);
  plugin->dtype_ = dtype_;
  return plugin;
}

// IPluginV3OneCore methods
nvinfer1::AsciiChar const* NormalizePlugin::getPluginName() const noexcept {
  return plugin_name_.c_str();
}

nvinfer1::AsciiChar const* NormalizePlugin::getPluginVersion() const noexcept {
  return plugin_version_.c_str();
}

nvinfer1::AsciiChar const* NormalizePlugin::getPluginNamespace() const noexcept {
  return plugin_namespace_.c_str();
}

// IPluginV3OneBuild methods
int32_t NormalizePlugin::getNbOutputs() const noexcept {
  return 1;
}

int32_t NormalizePlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  dtype_ = nvinfer1::DataType::kFLOAT;
  return 0;
}

int32_t NormalizePlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes,
    int32_t nbInputs) const noexcept {
  TORCHTRT_CHECK(nbOutputs == 1, "Expected 1 output");
  outputTypes[0] = nvinfer1::DataType::kFLOAT;
  return 0;
}

int32_t NormalizePlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs,
    nvinfer1::DimsExprs const* shapeInputs,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  
  TORCHTRT_CHECK(nbInputs == 1, "Expected 1 input");
  TORCHTRT_CHECK(nbOutputs == 1, "Expected 1 output");
  
  outputs[0].nbDims = keep_dims_ ? inputs[0].nbDims : inputs[0].nbDims - axes_.size();

  // For order-0 norm, when the norm dimension is None, it should normalize across all dimensions.
  bool isAxisNone = std::all_of(axes_.begin(), axes_.end(), [](int32_t i) { return i == 0; }) &&
      ((int32_t)axes_.size() == inputs[0].nbDims);
  if (isAxisNone) {
    std::iota(axes_.data(), axes_.data() + axes_.size(), 0);
  }
  
  int64_t out_idx = 0;
  for (int64_t i = 0; i < inputs[0].nbDims; i++) {
    if (std::find(axes_.begin(), axes_.end(), i) != axes_.end()) {
      if (keep_dims_) {
        outputs[0].d[out_idx] = exprBuilder.constant(1);
        out_idx += 1;
      }
    } else {
      if (!isAxisNone) {
        outputs[0].d[out_idx] = exprBuilder.constant(inputs[0].d[i]->getConstantValue());
      } else {
        outputs[0].d[out_idx] = exprBuilder.constant(1);
      }
      out_idx += 1;
    }
  }
  
  return 0;
}

bool NormalizePlugin::supportsFormatCombination(
    int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  
  if (pos < 0 || pos > 1) {
    LOG_ERROR("There should be exactly 2 connections to the plugin - 1 input, 1 output");
    return false;
  }
  if (nbInputs != 1) {
    LOG_ERROR("Expected a single tensor as input to normalize plugin");
    return false;
  }
  if (nbOutputs != 1) {
    LOG_ERROR("Expected a single tensor as output to normalize plugin");
    return false;
  }

  const nvinfer1::DynamicPluginTensorDesc& in = inOut[0];

  if (pos == 0) {
    return (in.desc.type == nvinfer1::DataType::kFLOAT) && (in.desc.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const nvinfer1::DynamicPluginTensorDesc& out = inOut[1];

  return (in.desc.type == out.desc.type) && (in.desc.format == out.desc.format);
}

size_t NormalizePlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const* inputs,
    int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

// IPluginV3OneRuntime methods
int32_t NormalizePlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* in,
    int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* out,
    int32_t nbOutputs) noexcept {
  return 0;
}

int32_t NormalizePlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs,
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

nvinfer1::IPluginV3* NormalizePlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept {
  return clone();
}

nvinfer1::PluginFieldCollection const* NormalizePlugin::getFieldsToSerialize() noexcept {
  return &mFCToSerialize;
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

nvinfer1::AsciiChar const* NormalizePluginCreator::getPluginNamespace() const noexcept {
  return plugin_namespace_.c_str();
}

nvinfer1::AsciiChar const* NormalizePluginCreator::getPluginName() const noexcept {
  return name_.c_str();
}

nvinfer1::AsciiChar const* NormalizePluginCreator::getPluginVersion() const noexcept {
  return plugin_version_.c_str();
}

nvinfer1::IPluginV3* NormalizePluginCreator::createPlugin(
    nvinfer1::AsciiChar const* name,
    nvinfer1::PluginFieldCollection const* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
  return new NormalizePlugin(fc);
}

nvinfer1::PluginFieldCollection const* NormalizePluginCreator::getFieldNames() noexcept {
  return &mFC;
}

REGISTER_TORCHTRT_PLUGIN(NormalizePluginCreator, "NormalizePlugin", "1");

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
