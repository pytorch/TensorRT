#ifndef TRTORCH_INTERPOLATE_PLUGIN_H
#define TRTORCH_INTERPOLATE_PLUGIN_H

#include <string>
#include <iostream>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <cudnn.h>

#include "core/util/prelude.h"
#include "torch/torch.h"
#include "NvInfer.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

class InterpolatePlugin : public nvinfer1::IPluginV2DynamicExt {
private:
    at::TensorOptions tensor_options_;
    DataType dtype_;

    std::vector<int64_t> in_shape_;
    std::vector<int64_t> out_shape_;
    std::vector<int64_t> size_;
    std::string mode_;
    bool align_corners_;

protected:
    // To prevent compiler warnings
    using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
    using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
    using nvinfer1::IPluginV2DynamicExt::supportsFormat;
    using nvinfer1::IPluginV2DynamicExt::configurePlugin;
    using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
    using nvinfer1::IPluginV2DynamicExt::enqueue;

public:
    InterpolatePlugin(std::vector<int64_t> in_shape, std::vector<int64_t> out_shape, std::vector<int64_t> size, std::string mode, bool align_corners);

    InterpolatePlugin(const char *data, size_t length);

    InterpolatePlugin() = delete;

    std::vector<int64_t> getInputShape();

    std::vector<int64_t> getOutputShape();

    std::vector<int64_t> getOutputSize();

    int getNbOutputs() const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    const char* getPluginNamespace() const override;

    void setPluginNamespace(const char* pluginNamespace) override {};

    nvinfer1::IPluginV2DynamicExt* clone() const override;

    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs, nvinfer1::IExprBuilder &exprBuilder) override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    int initialize() override;

    void terminate() override {}

    void serialize(void* buffer) const;

    std::string serializeToString() const;

    size_t getSerializationSize() const override;

    void destroy() override {}

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;

    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) override;

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const override;

    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void *const *inputs,
                                                                                                           void *const *outputs, void *workspace,
                                                                                                           cudaStream_t stream) override;
};

class InterpolatePluginCreator : public nvinfer1::IPluginCreator {
private:
    std::string name_;

public:
    InterpolatePluginCreator() = default;

    const char* getPluginNamespace() const override;

    void setPluginNamespace(const char* libNamespace) override {};

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection *fc) override;

    InterpolatePlugin* createPlugin(const char* name, std::vector<int64_t> in_shape, std::vector<int64_t> out_shape, std::vector<int64_t> size, std::string mode, bool align_corners);

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void *serialData, size_t serialLength) override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;
};

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

#endif // TRTORCH_INTERPOLATE_PLUGIN_H