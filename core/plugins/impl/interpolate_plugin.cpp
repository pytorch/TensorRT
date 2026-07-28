#include "core/plugins/impl/interpolate_plugin.h"
#include "core/plugins/plugins.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

/*
 * InterpolatePlugin class implementations
 */

InterpolatePlugin::InterpolatePlugin(
    std::vector<int64_t> in_shape,
    std::vector<int64_t> out_shape,
    std::vector<int64_t> size,
    std::vector<double> scales,
    std::string mode,
    bool align_corners,
    bool use_scales)
    : in_shape_(in_shape),
      out_shape_(out_shape),
      size_(size),
      scales_(scales),
      mode_(mode),
      align_corners_(align_corners),
      use_scales_(use_scales) {
  if (use_scales) {
    TORCHTRT_ASSERT(mode_ != "adaptive_avg_pool2d", "use_scales is not valid for adaptive_avg_pool2d");
    TORCHTRT_ASSERT(
        scales_.size() != 0, "Attempted to use interpolate plugin without providing scales while use_scales=true");
    at::Tensor input = at::randint(1, 10, in_shape, {at::kCUDA});
    at::Tensor output;

    if (mode_ == "linear") {
      output = at::upsample_linear1d(input, c10::nullopt, align_corners_, scales_[0]);
    } else if (mode_ == "bilinear") {
      output = at::upsample_bilinear2d(input, c10::nullopt, align_corners_, scales_);
      std::cout << output.sizes() << std::endl;
    } else if (mode_ == "trilinear") {
      output = at::upsample_trilinear3d(input, c10::nullopt, align_corners_, scales_);
    }

    out_shape_ = output.sizes().vec();
  } else {
    TORCHTRT_ASSERT(
        (size_.size() != 0 && out_shape_.size() != 0),
        "Attempted to use interpolate plugin without providing output size while use_scales=false");
  }
}

InterpolatePlugin::InterpolatePlugin(const char* data, size_t length) {
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
    input_archive.read("scales", value);
    scales_ = value.toDoubleVector();
  }
  {
    torch::IValue value;
    input_archive.read("mode", value);
    mode_ = value.toStringRef();
  }
  {
    torch::IValue value;
    input_archive.read("align_corners", value);
    align_corners_ = value.toBool();
  }
  {
    torch::IValue value;
    input_archive.read("use_scales", value);
    use_scales_ = value.toBool();
  }
}

std::vector<int64_t> InterpolatePlugin::getInputShape() {
  return in_shape_;
}

std::vector<int64_t> InterpolatePlugin::getOutputShape() {
  return out_shape_;
}

std::vector<int64_t> InterpolatePlugin::getOutputSize() {
  return size_;
}

// IPluginV3

nvinfer1::IPluginCapability* InterpolatePlugin::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
  switch (type) {
    case nvinfer1::PluginCapabilityType::kBUILD:
      return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
      return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    case nvinfer1::PluginCapabilityType::kCORE:
      return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    default:
      return nullptr;
  }
}

nvinfer1::IPluginV3* InterpolatePlugin::clone() noexcept {
  return new InterpolatePlugin(in_shape_, out_shape_, size_, scales_, mode_, align_corners_, use_scales_);
}

// IPluginV3OneCore

const char* InterpolatePlugin::getPluginName() const noexcept {
  return "Interpolate";
}

const char* InterpolatePlugin::getPluginVersion() const noexcept {
  return "1";
}

const char* InterpolatePlugin::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

// IPluginV3OneBuild

int32_t InterpolatePlugin::getNbOutputs() const noexcept {
  if (mode_ == "adaptive_max_pool2d") {
    return 2;
  } else {
    return 1;
  }
}

int32_t InterpolatePlugin::getOutputShapes(
    const nvinfer1::DimsExprs* inputs,
    int32_t nbInputs,
    const nvinfer1::DimsExprs* shapeInputs,
    int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs,
    int32_t nbOutputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  outputs[0] = inputs[0];

  // TODO: This should enable the case of using this plugin with dynamic shape, scale factor and align corners == true
  // to cover the different implementations between PyTorch and TRT. However TRT currently does not support doubles for
  // ExprBuilder constants. Once that is possible enable this code and remove the code in the constructor if
  // (use_scales_) {
  //   ...
  // } else {
  for (unsigned int i = 0; i < out_shape_.size(); i++) {
    outputs[0].d[i] = exprBuilder.constant(out_shape_[i]);
  }
  //}

  return 0;
}

int32_t InterpolatePlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes,
    int32_t nbOutputs,
    const nvinfer1::DataType* inputTypes,
    int32_t nbInputs) const noexcept {
  for (int32_t i = 0; i < nbOutputs; i++) {
    outputTypes[i] = nvinfer1::DataType::kFLOAT;
  }
  return 0;
}

bool InterpolatePlugin::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::DynamicPluginTensorDesc* inOut,
    int32_t nbInputs,
    int32_t nbOutputs) noexcept {
  if (nbInputs != 1) {
    LOG_ERROR("Expected a single tensor as input to interpolate plugin");
  }
  if (mode_ == "adaptive_max_pool2d") {
    if (nbOutputs != 2) {
      LOG_ERROR("Expected 2 tensors as output to interpolate plugin");
    }
    if (pos < 0 || pos > 2) {
      LOG_ERROR("There should be exactly 3 connections to the plugin - 1 input, 2 output");
    }
  } else {
    if (nbOutputs != 1) {
      LOG_ERROR("Expected a single tensor as output to interpolate plugin");
    }
    if (pos < 0 || pos > 1) {
      LOG_ERROR("There should be exactly 2 connections to the plugin - 1 input, 1 output");
    }
  }

  const nvinfer1::PluginTensorDesc& in = inOut[0].desc;

  if (pos == 0) {
    return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
  }

  // pos == 1, accessing information about output tensor
  const nvinfer1::PluginTensorDesc& out = inOut[1].desc;
  return (in.type == out.type) && (in.format == out.format);
}

int32_t InterpolatePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
  dtype_ = nvinfer1::DataType::kFLOAT;
  return 0;
}

size_t InterpolatePlugin::getWorkspaceSize(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int32_t nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int32_t nbOutputs) const noexcept {
  return 0;
}

std::string InterpolatePlugin::serializeToString() const {
  torch::serialize::OutputArchive output_archive;

  output_archive.write("in_shape", torch::IValue(in_shape_));
  output_archive.write("out_shape", torch::IValue(out_shape_));
  output_archive.write("size", torch::IValue(size_));
  output_archive.write("scales", torch::IValue(scales_));
  output_archive.write("mode", torch::IValue(mode_));
  output_archive.write("align_corners", torch::IValue(align_corners_));
  output_archive.write("use_scales", torch::IValue(use_scales_));

  std::ostringstream data_str;
  output_archive.save_to(data_str);
  return data_str.str();
}

nvinfer1::PluginFieldCollection const* InterpolatePlugin::getFieldsToSerialize() noexcept {
  mSerializedData = serializeToString();
  mSerializationFields.clear();
  mSerializationFields.emplace_back(
      "data", mSerializedData.data(), nvinfer1::PluginFieldType::kCHAR, mSerializedData.size());
  mSerializationFC.nbFields = static_cast<int32_t>(mSerializationFields.size());
  mSerializationFC.fields = mSerializationFields.data();
  return &mSerializationFC;
}

// IPluginV3OneRuntime

int32_t InterpolatePlugin::onShapeChange(
    const nvinfer1::PluginTensorDesc* in,
    int32_t nbInputs,
    const nvinfer1::PluginTensorDesc* out,
    int32_t nbOutputs) noexcept {
  return 0;
}

int32_t InterpolatePlugin::enqueue(
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
  at::Tensor out;
  if (use_scales_) {
    if (mode_ == "linear") {
      out = at::upsample_linear1d(input, c10::nullopt, align_corners_, {scales_[0]});
    } else if (mode_ == "bilinear") {
      out = at::upsample_bilinear2d(input, c10::nullopt, align_corners_, scales_);
    } else if (mode_ == "trilinear") {
      out = at::upsample_trilinear3d(input, c10::nullopt, align_corners_, scales_);
    }
  } else {
    if (mode_ == "linear") {
      out = at::upsample_linear1d(input, {size_[0]}, align_corners_);
    } else if (mode_ == "bilinear") {
      out = at::upsample_bilinear2d(input, {size_[0], size_[1]}, align_corners_);
    } else if (mode_ == "trilinear") {
      out = at::upsample_trilinear3d(input, {size_[0], size_[1], size_[2]}, align_corners_);
    } else if (mode_ == "adaptive_avg_pool1d") {
      out = at::adaptive_avg_pool1d(input, {size_[0]});
    } else if (mode_ == "adaptive_max_pool1d") {
      out = std::get<0>(at::adaptive_max_pool1d(input, {size_[0]}));
    } else if (mode_ == "adaptive_avg_pool2d") {
      out = at::adaptive_avg_pool2d(input, {size_[0], size_[1]});
    } else if (mode_ == "adaptive_max_pool2d") {
      out = std::get<0>(at::adaptive_max_pool2d(input, {size_[0], size_[1]}));
    } else if (mode_ == "adaptive_avg_pool3d") {
      out = at::adaptive_avg_pool3d(input, {size_[0], size_[1], size_[2]});
    } else if (mode_ == "adaptive_max_pool3d") {
      out = std::get<0>(at::adaptive_max_pool3d(input, {size_[0], size_[1], size_[2]}));
    }
  }

  output.copy_(out);
  cudaEvent_t torch_event;
  cudaEventCreate(&torch_event);
  cudaEventRecord(torch_event, torch_stream.stream());

  cudaStreamWaitEvent(stream, torch_event, 0);

  cudaEventDestroy(event);
  cudaEventDestroy(torch_event);

  return 0;
}

nvinfer1::IPluginV3* InterpolatePlugin::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept {
  return new InterpolatePlugin(in_shape_, out_shape_, size_, scales_, mode_, align_corners_, use_scales_);
}

/*
 * InterpolatePluginCreator class implementations
 */

InterpolatePluginCreator::InterpolatePluginCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("in_shape", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("out_shape", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("out_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("scales", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("mode", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("align_corners", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("use_scales", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* InterpolatePluginCreator::getPluginNamespace() const noexcept {
  return "torch_tensorrt";
}

const char* InterpolatePluginCreator::getPluginName() const noexcept {
  return "Interpolate";
}

const char* InterpolatePluginCreator::getPluginVersion() const noexcept {
  return "1";
}

nvinfer1::IPluginV3* InterpolatePluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fc,
    nvinfer1::TensorRTPhase phase) noexcept {
  if (phase == nvinfer1::TensorRTPhase::kBUILD) {
    std::vector<int64_t> in_shape;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> out_size;
    std::vector<double> scales;
    std::string mode;
    int32_t align_corners = 0;
    int32_t use_scales = 0;

    for (int i = 0; i < fc->nbFields; i++) {
      std::string field_name(fc->fields[i].name);
      if (field_name.compare("in_shape") == 0) {
        auto in_shape_values = static_cast<const int32_t*>(fc->fields[i].data);
        in_shape.assign(in_shape_values, in_shape_values + fc->fields[i].length);
      } else if (field_name.compare("out_shape") == 0) {
        auto out_shape_values = static_cast<const int32_t*>(fc->fields[i].data);
        out_shape.assign(out_shape_values, out_shape_values + fc->fields[i].length);
      } else if (field_name.compare("out_size") == 0) {
        auto out_size_values = static_cast<const int32_t*>(fc->fields[i].data);
        out_size.assign(out_size_values, out_size_values + fc->fields[i].length);
      } else if (field_name.compare("scales") == 0) {
        auto scales_values = static_cast<const double*>(fc->fields[i].data);
        scales.assign(scales_values, scales_values + fc->fields[i].length);
      } else if (field_name.compare("mode") == 0) {
        mode = *static_cast<const std::string*>(fc->fields[i].data);
      } else if (field_name.compare("align_corners") == 0) {
        align_corners = *static_cast<const int32_t*>(fc->fields[i].data);
      } else if (field_name.compare("use_scales") == 0) {
        use_scales = *static_cast<const int32_t*>(fc->fields[i].data);
      }
    }
    return new InterpolatePlugin(in_shape, out_shape, out_size, scales, mode, (bool)align_corners, (bool)use_scales);
  } else { // TensorRTPhase::kRUNTIME - deserialization
    auto const* data = static_cast<const char*>(fc->fields[0].data);
    size_t length = fc->fields[0].length;
    return new InterpolatePlugin(data, length);
  }
}

const nvinfer1::PluginFieldCollection* InterpolatePluginCreator::getFieldNames() noexcept {
  return &mFC;
}

REGISTER_TORCHTRT_PLUGIN_V3(InterpolatePluginCreator);

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
