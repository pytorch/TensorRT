#include "interpolate_plugin.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

/*
 * InterpolatePlugin class implementations
 */

InterpolatePlugin::InterpolatePlugin(std::vector<int64_t> in_shape, std::vector<int64_t> out_shape, std::vector<int64_t> size, std::string mode, bool align_corners) :
    in_shape_(in_shape), out_shape_(out_shape), size_(size), mode_(mode), align_corners_(align_corners)
{}

InterpolatePlugin::InterpolatePlugin(const char *data, size_t length) {
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
    {
        torch::IValue value;
        input_archive.read("align_corners", value);
        align_corners_ = value.toBool();
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

int InterpolatePlugin::getNbOutputs() const {
    return 1;
}

const char* InterpolatePlugin::getPluginType() const {
    return "Interpolate";
}

const char* InterpolatePlugin::getPluginVersion() const{
    return "1";
}

const char* InterpolatePlugin::getPluginNamespace() const {
    return "";
}


nvinfer1::IPluginV2DynamicExt* InterpolatePlugin::clone() const {
    return new InterpolatePlugin(in_shape_, out_shape_, size_, mode_, align_corners_);
}

nvinfer1::DimsExprs InterpolatePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) {
   nvinfer1::DimsExprs output(inputs[0]);

   for (unsigned int i = 0; i < out_shape_.size(); i++) {
       output.d[i] = exprBuilder.constant(out_shape_[i]);
   }

   return output;
}

nvinfer1::DataType InterpolatePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
    return DataType::kFLOAT;
}

int InterpolatePlugin::initialize() {
    tensor_options_ = tensor_options_.device(c10::kCPU);

    // c10::kFloat = FLOAT32
    tensor_options_ = tensor_options_.dtype(c10::kFloat);

    return 0;
}


void InterpolatePlugin::serialize(void* buffer) const {
    std::string data = serializeToString();
    size_t size = getSerializationSize();

    data.copy((char*) buffer, size);
}

std::string InterpolatePlugin::serializeToString() const {
    torch::serialize::OutputArchive output_archive;

    output_archive.write("in_shape", torch::IValue(in_shape_));
    output_archive.write("out_shape", torch::IValue(out_shape_));
    output_archive.write("size", torch::IValue(size_));
    output_archive.write("mode", torch::IValue(mode_));
    output_archive.write("align_corners", torch::IValue(align_corners_));

    std::ostringstream data_str;
    output_archive.save_to(data_str);

    return data_str.str();
}

size_t InterpolatePlugin::getSerializationSize() const {
    return serializeToString().size();
}

bool InterpolatePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    TRTORCH_ASSERT(0 <= pos && pos <= 1, "There should be exactly 2 connections to the plugin - 1 input, 1 output");
    TRTORCH_ASSERT(nbInputs == 1, "Expected a single tensor as input to interpolate plugin");
    TRTORCH_ASSERT(nbOutputs == 1, "Expected a single tensor as output to interpolate plugin");

    const PluginTensorDesc& in = inOut[0];

    if (pos == 0) {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == nvinfer1::TensorFormat::kLINEAR);
    }

    // pos == 1, accessing information about output tensor
    const PluginTensorDesc& out = inOut[1];

    return (in.type == out.type) && (in.format == out.format);
}

void InterpolatePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
    dtype_ = DataType::kFLOAT;
}

size_t InterpolatePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
    return 0;
}

int InterpolatePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
                                                                                                        void* const* outputs, void* workspace,
                                                                                                        cudaStream_t stream) {
    // TODO: When PyTorch updates to cuDNN 8 try moving back to CUDA based ATen kernels
    // HACK: WAR because there is a segfault if you try to create a CUDA Tensor in the context of TensorRT execution
    float* input_blob = (float*) malloc(util::volume(inputDesc->dims) * sizeof(float));
    cudaMemcpyAsync(input_blob, static_cast<const void*>(inputs[0]), util::volume(inputDesc->dims) * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    at::Tensor input = at::from_blob((void*)input_blob, util::toVec(inputDesc->dims), tensor_options_);

    at::Tensor output;
    if (mode_ == "adaptive_pool2d") {
        output = at::adaptive_avg_pool2d(input, {size_[0], size_[1]});
    }

    output = output.contiguous();
    for (int i = 0; i < util::volume(outputDesc->dims); i++) {
        std::cout << ((float*)output.data_ptr())[i] << std::endl;
    }

    cudaMemcpyAsync(outputs[0], output.data_ptr(), util::volume(outputDesc->dims) * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    free(input_blob);

    return 0;
}

/*
 * InterpolatePluginCreator class implementations
 */
const char* InterpolatePluginCreator::getPluginNamespace() const {
    return "";
}

const char* InterpolatePluginCreator::getPluginName() const {
    return "Interpolate";
}

const char* InterpolatePluginCreator::getPluginVersion() const {
    return "1";
}

nvinfer1::IPluginV2* InterpolatePluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) {
    return nullptr;
}

InterpolatePlugin* InterpolatePluginCreator::createPlugin(const char* name, std::vector<int64_t> in_shape, std::vector<int64_t> out_shape,
                                                                                                           std::vector<int64_t> size,
                                                                                                           std::string mode, bool align_corners) {
    name_ = name;
    return new InterpolatePlugin(in_shape, out_shape, size, mode, align_corners);
}

nvinfer1::IPluginV2* InterpolatePluginCreator::deserializePlugin(const char* name, const void *serialData, size_t serialLength) {
    name_ = name;
    return new InterpolatePlugin((const char*) serialData, serialLength);
}

const nvinfer1::PluginFieldCollection* InterpolatePluginCreator::getFieldNames() {
    return nullptr;
}

REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch