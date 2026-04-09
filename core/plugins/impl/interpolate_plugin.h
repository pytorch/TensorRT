#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
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

class InterpolatePlugin : public nvinfer1::IPluginV3,
                          public nvinfer1::IPluginV3OneBuild,
                          public nvinfer1::IPluginV3OneRuntime,
                          public nvinfer1::IPluginV3OneCore {
 private:
  nvinfer1::DataType dtype_;

  std::vector<int64_t> in_shape_;
  std::vector<int64_t> out_shape_;
  std::vector<int64_t> size_;
  std::vector<double> scales_;
  std::string mode_;
  bool align_corners_;
  bool use_scales_;

  std::string serializeToString() const;

  // For getFieldsToSerialize()
  mutable std::string mSerializedData;
  mutable std::vector<nvinfer1::PluginField> mSerializationFields;
  mutable nvinfer1::PluginFieldCollection mSerializationFC{};

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

  // IPluginV3
  nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
  nvinfer1::IPluginV3* clone() noexcept override;

  // IPluginV3OneCore
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const char* getPluginNamespace() const noexcept override;

  // IPluginV3OneBuild
  int32_t getNbOutputs() const noexcept override;
  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      int32_t nbInputs,
      const nvinfer1::DimsExprs* shapeInputs,
      int32_t nbShapeInputs,
      nvinfer1::DimsExprs* outputs,
      int32_t nbOutputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  int32_t getOutputDataTypes(
      nvinfer1::DataType* outputTypes,
      int32_t nbOutputs,
      const nvinfer1::DataType* inputTypes,
      int32_t nbInputs) const noexcept override;
  bool supportsFormatCombination(
      int32_t pos,
      const nvinfer1::DynamicPluginTensorDesc* inOut,
      int32_t nbInputs,
      int32_t nbOutputs) noexcept override;
  int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int32_t nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int32_t nbOutputs) noexcept override;
  size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      int32_t nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      int32_t nbOutputs) const noexcept override;

  // IPluginV3OneRuntime
  nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;
  int32_t onShapeChange(
      const nvinfer1::PluginTensorDesc* in,
      int32_t nbInputs,
      const nvinfer1::PluginTensorDesc* out,
      int32_t nbOutputs) noexcept override;
  int32_t enqueue(
      const nvinfer1::PluginTensorDesc* inputDesc,
      const nvinfer1::PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;
  nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
};

class InterpolatePluginCreator : public nvinfer1::IPluginCreatorV3One {
 private:
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  nvinfer1::PluginFieldCollection mFC;

 public:
  InterpolatePluginCreator();

  const char* getPluginNamespace() const noexcept override;
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  nvinfer1::IPluginV3* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc,
      nvinfer1::TensorRTPhase phase) noexcept override;
};

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
