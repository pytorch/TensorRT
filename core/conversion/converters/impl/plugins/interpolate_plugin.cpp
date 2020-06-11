#include <torch/extension.h>
#include <torch/script.h>
#include <string>
#include <iostream>
#include <sstream>
#include <NvInfer.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include "NvInferVersion.h"
#include <vector>
#include <cudnn.h>
#include <NVInferRuntime.h>
#include <NVInferRuntimeCommon.h>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {
namespace {

class InterpolatePlugin : public nvinfer1::IPluginV2DynamicExt {
private:
    at::TensorOptions tensor_options;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_sizes;
    DataType dtype;

    std::vector<int64_t> size;
    std::string mode;
    bool align_corners;

public:
    InterpolatePlugin(const char* name, std::vector<int64_t> in_shape, 
                                        std::vector<int64_t> out_shape, 
                                        std::string mode, 
                                        bool align_corners) : name(name), in_shape(in_shape), out_shape(out_shape), mode(mode), align_corners(align_corners) {}



    const char* getPluginType() const override {
        return "Interpolate";
    }

    const char* getPluginVersion() const override {
        return "1";
    }

    const char* getPluginNamespace() const override {
        return "trtorch";
    }

    void setPluginNamespace(const char* pluginNamespace) {}

    int getTensorRTVersion() const override {
        return NV_TENSORRT_MAJOR;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        auto* plugin = new InterpolatePlugin(*this);
        return plugin;
    }

    nvinfer::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) const override {
       
    }

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {

    }

    int getNbOutputs() const override {
        return 1;
    }

    int initialize() override {

    }

    void terminate() override {

    }

    void serialize(void* buffer) const override {

    }

    void size_t getSerializationSize() const override {

    }

    void destroy() override {

    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override {

    }

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override {

    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override {

    }

    void attachToContext(nvinfer1::cudnnContext*, nvinfer1::cublasContext*, nvinfer1::IGpuAllocator*) override {}

    void detachFromContext() override {}

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void *const *inputs, 
                                                                                                           void *const *outputs, void *workspace, 
                                                                                                           cudaStream_t stream) override {
                                                                                                               
    }




private:
    std::string name;
    std::vector<int64_t> in_shape;
    std::vector<int64_t> out_shape;
    std::string mode;
    bool align_corners;

    nvinfer1::DataType dtype;
}


class InterpolatePluginCreator : public nvinfer1::IPluginCreator {
public:
    InterpolatePluginCreator() {}

    int getTensorRTVersion() const override {
        return NV_TENSORRT_MAJOR;
    }

    const char *getPluginNamespace() const override {
        return "trtorch";
    }

    void setPluginNamespace(const char* libNamespace) override {}
    
    const char *getPluginName() const override {
        return "interpolate";
    }

    const char *getPluginVersion() const override {
        return "1";
    }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) override {
        return nullptr;
    }

    nvinfer1::IPluginV2* createPlugin(const char* name, std::vector<int64_t> in_shape, std::vector<int64_t> out_shape, std::string mode, bool align_corners) {
        return new InterpolatePlugin(name, in_shape, out_shape, mode, align_corners);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void *serialData, size_t serialLength) override {
        return nullptr;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        return nullptr;
    }
}

REGISTER_TENSORRT_PLUGIN(InterpolatePluginCreator);

} // namespace
} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

