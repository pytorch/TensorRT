#pragma once

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

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

class InterpolatePlugin : public nvinfer1::IPluginV2DynamicExt {
 private:
  nvinfer1::DataType dtype_;

  std::vector<int64_t> in_shape_;
  std::vector<int64_t> out_shape_;
  std::vector<int64_t> size_;
  std::vector<double> scales_;
  std::string mode_;
  bool align_corners_;
  bool use_scales_;

 public:
  InterpolatePlugin(
      std::vector<int64_t> in_shape,
      std::vector<int64_t> out_shape,
      std::vector<int64_t> size,
      std::vector<double> scales,
      std::string mode,
      bool align_corners,
      bool use_scales);

  InterpolatePlugin(const char* data, size_t length);

  InterpolatePlugin() = delete;

  std::vector<int64_t> getInputShape();

  std::vector<int64_t> getOutputShape();

  std::vector<int64_t> getOutputSize();

  int getNbOutputs() const noexcept override;

  const char* getPluginType() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  const char* getPluginNamespace() const noexcept override;

  void setPluginNamespace(const char* pluginNamespace) noexcept override{};

  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs)
      const noexcept override;

  int initialize() noexcept override;

  void terminate() noexcept override {}

  void serialize(void* buffer) const noexcept override;

  std::string serializeToString() const;

  size_t getSerializationSize() const noexcept override;

  void destroy() noexcept override {}

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
      override;

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int nbOutputs) noexcept override;

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const noexcept override;

  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
};

class InterpolatePluginCreator : public nvinfer1::IPluginCreator {
 private:
  std::string name_;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  nvinfer1::PluginFieldCollection mFC;

 public:
  InterpolatePluginCreator();

  const char* getPluginNamespace() const noexcept override;

  void setPluginNamespace(const char* libNamespace) noexcept override{};

  const char* getPluginName() const noexcept override;

  const char* getPluginVersion() const noexcept override;

  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
      override;

  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
};

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
