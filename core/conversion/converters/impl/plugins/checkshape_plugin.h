#ifndef _CHECKSHAPE_PLUGIN_H_
#define _CHECKSHAPE_PLUGIN_H_

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <sstream>
// #include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>

using std::string;
using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

class CheckShapePlugin: public IPluginV2DynamicExt {
  public:
    // constructor
    CheckShapePlugin(const std::string name, int in_rank, int expand_rank);
    CheckShapePlugin(const std::string name, const void * data, size_t length);
    CheckShapePlugin(const CheckShapePlugin &obj);
    CheckShapePlugin() = delete; // delete default constructor

    // destructor
    ~CheckShapePlugin() override;// = default;

    // public member functions
    // inherited from IPluginV2
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int32_t getNbOutputs() const override;
    int32_t initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void *buffer) const override;
    void destroy() override;
    void setPluginNamespace(const char *pluginNamespace) override;
    const char* getPluginNamespace() const override;

    // inherited from IPluginV2Ext
    DataType getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const override;
    void attachToContext(cudnnContext *, cublasContext *, IGpuAllocator *) override;
    void detachFromContext() override;

    // others
    IPluginV2DynamicExt* clone() const override;
    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) override;
    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) override;
    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) override;
    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const override;
    int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

  private:
    int inputRank;
    int expandRank;
    string mLayerName;
    string mPluginNamespace;
};


class CheckShapePluginCreator : public IPluginCreator {
  public:
    // constructor
    CheckShapePluginCreator();
    // destructor
    ~CheckShapePluginCreator() override = default;

    // public member functions
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;
    IPluginV2* createPlugin(const char *name, const PluginFieldCollection *fc) override;
    IPluginV2* deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;
    void setPluginNamespace(const char *pluginNamespace) override;
    const char* getPluginNamespace() const override;
  private:
    string mPluginNamespace;
};

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

#endif // _CHECKSHAPE_PLUGIN_H_