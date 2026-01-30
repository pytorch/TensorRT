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
                          public nvinfer1::IPluginV3OneCore,
                          public nvinfer1::IPluginV3OneBuild,
                          public nvinfer1::IPluginV3OneRuntime {
 private:
  nvinfer1::DataType dtype_;

  std::vector<int64_t> in_shape_;
  std::vector<int64_t> out_shape_;
  std::vector<int64_t> size_;
  std::vector<double> scales_;
  std::string mode_;
  bool align_corners_;
  bool use_scales_;
  
  // Serialization fields
  mutable std::string mSerializedData;
  mutable std::vector<nvinfer1::PluginField> mSerializationFields;
  mutable nvinfer1::PluginFieldCollection mSerializationFC;

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

  // IPluginV3 methods
  nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;

  nvinfer1::IPluginV3* clone() noexcept override;

  // IPluginV3OneCore methods
  char const* getPluginName() const noexcept override;

  char const* getPluginVersion() const noexcept override;

  char const* getPluginNamespace() const noexcept override;

  // IPluginV3OneBuild methods
  int32_t getNbOutputs() const noexcept override;

  int32_t configurePlugin(
      nvinfer1::DynamicPluginTensorDesc const* in,
      int32_t nbInputs,
      nvinfer1::DynamicPluginTensorDesc const* out,
      int32_t nbOutputs) noexcept override;

  int32_t getOutputDataTypes(
      nvinfer1::DataType* outputTypes,
      int32_t nbOutputs,
      nvinfer1::DataType const* inputTypes,
      int32_t nbInputs) const noexcept override;

  int32_t getOutputShapes(
      nvinfer1::DimsExprs const* inputs,
      int32_t nbInputs,
      nvinfer1::DimsExprs const* shapeInputs,
      int32_t nbShapeInputs,
      nvinfer1::DimsExprs* outputs,
      int32_t nbOutputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  bool supportsFormatCombination(
      int32_t pos,
      nvinfer1::DynamicPluginTensorDesc const* inOut,
      int32_t nbInputs,
      int32_t nbOutputs) noexcept override;

  size_t getWorkspaceSize(
      nvinfer1::DynamicPluginTensorDesc const* inputs,
      int32_t nbInputs,
      nvinfer1::DynamicPluginTensorDesc const* outputs,
      int32_t nbOutputs) const noexcept override;

  int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;

  int32_t getNbTactics() noexcept override;

  int32_t getFormatCombinationLimit() noexcept override;

  char const* getMetadataString() noexcept override;

  // IPluginV3OneRuntime methods
  int32_t setTactic(int32_t tactic) noexcept override;

  int32_t onShapeChange(
      nvinfer1::PluginTensorDesc const* in,
      int32_t nbInputs,
      nvinfer1::PluginTensorDesc const* out,
      int32_t nbOutputs) noexcept override;

  int32_t enqueue(
      nvinfer1::PluginTensorDesc const* inputDesc,
      nvinfer1::PluginTensorDesc const* outputDesc,
      void const* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;

  nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;

  nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

  // Serialization
  size_t getSerializationSize() const noexcept;

  void serialize(void* buffer) const noexcept;

  std::string serializeToString() const;
};

class InterpolatePluginCreator : public nvinfer1::IPluginCreatorV3One {
 private:
  std::string name_;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  nvinfer1::PluginFieldCollection mFC;

 public:
  InterpolatePluginCreator();

  char const* getPluginNamespace() const noexcept override;

  char const* getPluginName() const noexcept override;

  char const* getPluginVersion() const noexcept override;

  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

  nvinfer1::IPluginV3* createPlugin(
      char const* name,
      nvinfer1::PluginFieldCollection const* fc,
      nvinfer1::TensorRTPhase phase) noexcept override;
};

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
