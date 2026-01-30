#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

class NormalizePlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneBuild,
                        public nvinfer1::IPluginV3OneRuntime {
 private:
  std::string plugin_name_{"NormalizePlugin"};
  std::string plugin_namespace_{"torch_tensorrt"};
  std::string plugin_version_{"1"};
  
  nvinfer1::DataType dtype_;
  int32_t order_;
  std::vector<int32_t> axes_;
  int32_t keep_dims_;
  
  // Serialization fields
  std::vector<nvinfer1::PluginField> mDataToSerialize;
  nvinfer1::PluginFieldCollection mFCToSerialize;

 public:
  NormalizePlugin(int32_t order, std::vector<int32_t> axes, int32_t keep_dims);

  NormalizePlugin(const nvinfer1::PluginFieldCollection* fc);

  NormalizePlugin() = delete;

  // IPluginV3 methods
  nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
  
  nvinfer1::IPluginV3* clone() noexcept override;

  // IPluginV3OneCore methods  
  nvinfer1::AsciiChar const* getPluginName() const noexcept override;

  nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;

  nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

  // IPluginV3OneBuild methods
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

  int32_t getNbOutputs() const noexcept override;

  size_t getWorkspaceSize(
      nvinfer1::DynamicPluginTensorDesc const* inputs,
      int32_t nbInputs,
      nvinfer1::DynamicPluginTensorDesc const* outputs,
      int32_t nbOutputs) const noexcept override;

  // IPluginV3OneRuntime methods
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

 protected:
  // Helper to setup serialization
  void setupSerialization();
};

class NormalizePluginCreator : public nvinfer1::IPluginCreatorV3One {
 private:
  std::string name_{"NormalizePlugin"};
  std::string plugin_namespace_{"torch_tensorrt"};
  std::string plugin_version_{"1"};
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  nvinfer1::PluginFieldCollection mFC;

 public:
  NormalizePluginCreator();

  nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

  nvinfer1::AsciiChar const* getPluginName() const noexcept override;

  nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;

  nvinfer1::IPluginV3* createPlugin(
      nvinfer1::AsciiChar const* name,
      nvinfer1::PluginFieldCollection const* fc,
      nvinfer1::TensorRTPhase phase) noexcept override;

  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
};

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
