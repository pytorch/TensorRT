#ifndef TRTORCH_INTERPOLATE_PLUGIN_H
#define TRTORCH_INTERPOLATE_PLUGIN_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

using namespace nvinfer1;

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace plugins {

class CumsumPlugin : public nvinfer1::IPluginV2DynamicExt {
 private:
  at::TensorOptions tensor_options_;
  int64_t dim_;

 protected:
  // To prevent compiler warnings
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;

 public:
  CumsumPlugin(int dim);

  CumsumPlugin(const char* data, size_t length);

  CumsumPlugin() = delete;

  int getNbOutputs() const override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  const char* getPluginNamespace() const override;

  void setPluginNamespace(const char* pluginNamespace) override{};

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

  int initialize() override;

  void terminate() override {}

  void serialize(void* buffer) const;

  std::string serializeToString() const;

  size_t getSerializationSize() const override;

  void destroy() override {}

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)
      override;

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) override;

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const override;

  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) override;
};

class CumsumPluginCreator : public nvinfer1::IPluginCreator {
 private:
  std::string name_;

 public:
  CumsumPluginCreator() = default;

  const char* getPluginNamespace() const override;

  void setPluginNamespace(const char* libNamespace) override{};

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  CumsumPlugin* createPlugin(const char* name, int dim);

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

  const nvinfer1::PluginFieldCollection* getFieldNames() override;
};

} // namespace plugins
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

#endif // TRTORCH_INTERPOLATE_PLUGIN_H