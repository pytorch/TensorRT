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
    in_shape(in_shape), out_shape(out_shape), size(size), mode(mode), align_corners(align_corners) 
{}

InterpolatePlugin::InterpolatePlugin(const char *data, size_t length) {
    std::istringstream data_stream(std::string(data, length));
    
    torch::serialize::InputArchive input_archive;
    input_archive.load_from(data_stream);
    
    {
        torch::IValue value;
        input_archive.read("in_shape", value);
        in_shape = value.toIntVector();
    }
    {
        torch::IValue value;
        input_archive.read("out_shape", value);
        out_shape = value.toIntVector();
    }
    {
        torch::IValue value;
        input_archive.read("size", value);
        size = value.toIntVector();
    }
    {
        torch::IValue value;
        input_archive.read("mode", value);
        mode = value.toStringRef();
    }
    {
        torch::IValue value;
        input_archive.read("align_corners", value);
        align_corners = value.toBool();
    }
}

std::vector<int64_t> InterpolatePlugin::getInputShape() {
    return in_shape;
}

std::vector<int64_t> InterpolatePlugin::getOutputShape() {
    return out_shape;
}

std::vector<int64_t> InterpolatePlugin::getOutputSize() {
    return size;
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
    return new InterpolatePlugin(in_shape, out_shape, size, mode, align_corners);
}

nvinfer1::DimsExprs InterpolatePlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) {
   nvinfer1::DimsExprs output(inputs[0]);

   for (unsigned int i = 0; i < out_shape.size(); i++) {
       output.d[i] = exprBuilder.constant(out_shape[i]);
   }

   return output;
}

nvinfer1::DataType InterpolatePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
    return DataType::kFLOAT;
}

int InterpolatePlugin::initialize() {
    tensor_options = tensor_options.device(c10::kCUDA);

    // c10::kFloat = FLOAT32
    tensor_options = tensor_options.dtype(c10::kFloat);

    return 0;
}


void InterpolatePlugin::serialize(void* buffer) const {
    std::string data = serializeToString();
    size_t size = getSerializationSize();

    data.copy((char*) buffer, size);
}

std::string InterpolatePlugin::serializeToString() const {
    torch::serialize::OutputArchive output_archive;

    output_archive.write("in_shape", torch::IValue(in_shape));
    output_archive.write("out_shape", torch::IValue(out_shape));
    output_archive.write("size", torch::IValue(size));
    output_archive.write("mode", torch::IValue(mode));
    output_archive.write("align_corners", torch::IValue(align_corners));

    std::ostringstream data_str;
    output_archive.save_to(data_str);

    return data_str.str();
}

size_t InterpolatePlugin::getSerializationSize() const {
    return serializeToString().size();
}

bool InterpolatePlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    if (inOut->format != nvinfer1::TensorFormat::kLINEAR) {
        return false;
    } 

    if (inOut->type == DataType::kINT32 || inOut->type == DataType::kINT8) {
        return false;
    }

    return true;
}

void InterpolatePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
    dtype = DataType::kFLOAT;
}

size_t InterpolatePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
    return 0;
}

int InterpolatePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void *const *inputs, 
                                                                                                        void *const *outputs, void *workspace, 
                                                                                                        cudaStream_t stream) {

    at::Tensor input = at::from_blob((void*) inputs[0], in_shape, [](void*){}, tensor_options);
    at::Tensor output = at::from_blob(outputs[0], out_shape, [](void*){}, tensor_options);

    at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromPool();
    at::cuda::CUDAStreamGuard torch_guard(torch_stream);

    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, stream);

    cudaStreamWaitEvent(torch_stream.stream(), event, 0);

    if (mode == "linear") {
        at::upsample_linear1d_out(output, input, {size[0]}, align_corners);
    } else if (mode == "bilinear") {
        at::upsample_bilinear2d_out(output, input, {size[0], size[1]}, align_corners);
    } else if (mode == "trilinear") {
        at::upsample_trilinear3d_out(output, input, {size[0], size[1], size[2]}, align_corners);
    } else if (mode == "adaptive_pool2d") {
        at::adaptive_avg_pool2d_out(output, input, {size[0], size[1]});
    }

    cudaEvent_t torch_event;
    cudaEventCreate(&torch_event);
    cudaEventRecord(torch_event, torch_stream.stream());

    cudaStreamWaitEvent(stream, torch_event, 0);

    cudaEventDestroy(event);
    cudaEventDestroy(torch_event);

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
    name = name;
    return new InterpolatePlugin(in_shape, out_shape, size, mode, align_corners);
}

nvinfer1::IPluginV2* InterpolatePluginCreator::deserializePlugin(const char* name, const void *serialData, size_t serialLength) {
    name = name;
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