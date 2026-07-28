#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferPluginBase.h"
#include "NvInferRuntime.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace plugins {
namespace impl {

// ---------------------------------------------------------------------------
// ScatterAddPlugin
//
// Implements aten::index_put(src, indices_list, values, accumulate=True) as a
// TensorRT IPluginV3 plugin.  Uses ATen's CUDA scatter-add kernel directly in
// enqueue() — same code path as PyTorch eager, no Python overhead.
//
// Inputs  (2 + N, where N >= 1 is the number of index tensors):
//   0       – src           : (*src_shape)       float32 | float16 | bfloat16
//   1..N    – indices[0..N-1]: (P,)              int32 | int64
//   N+1     – values        : (*val_shape)        same dtype as src
//
// Output (1):
//   0 – out : same shape and dtype as src
// ---------------------------------------------------------------------------

class ScatterAddPlugin : public nvinfer1::IPluginV3,
                         public nvinfer1::IPluginV3OneCore,
                         public nvinfer1::IPluginV3OneBuild,
                         public nvinfer1::IPluginV3OneRuntime {
 public:
  ScatterAddPlugin();
  ScatterAddPlugin(const ScatterAddPlugin&) = default;
  ScatterAddPlugin& operator=(const ScatterAddPlugin&) = delete;
  ~ScatterAddPlugin() override = default;

  // ---- IPluginV3 -----------------------------------------------------------
  nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
  nvinfer1::IPluginV3* clone() noexcept override;

  // ---- IPluginV3OneCore ----------------------------------------------------
  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const char* getPluginNamespace() const noexcept override;

  // ---- IPluginV3OneBuild ---------------------------------------------------
  int32_t configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* in,
      int32_t nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* out,
      int32_t nbOutputs) noexcept override;

  int32_t getOutputDataTypes(
      nvinfer1::DataType* outputTypes,
      int32_t nbOutputs,
      const nvinfer1::DataType* inputTypes,
      int32_t nbInputs) const noexcept override;

  int32_t getOutputShapes(
      const nvinfer1::DimsExprs* inputs,
      int32_t nbInputs,
      const nvinfer1::DimsExprs* shapeInputs,
      int32_t nbShapeInputs,
      nvinfer1::DimsExprs* outputs,
      int32_t nbOutputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  bool supportsFormatCombination(
      int32_t pos,
      const nvinfer1::DynamicPluginTensorDesc* inOut,
      int32_t nbInputs,
      int32_t nbOutputs) noexcept override;

  int32_t getNbOutputs() const noexcept override;
  size_t getWorkspaceSize(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      int32_t nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      int32_t nbOutputs) const noexcept override;

  // ---- IPluginV3OneRuntime -------------------------------------------------
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
  const nvinfer1::PluginFieldCollection* getFieldsToSerialize() noexcept override;

 private:
  // Captured at configurePlugin / onShapeChange time.
  // Layout: input[0]=src, input[1..n_indices_]=indices, input[n_indices_+1]=values
  nvinfer1::DataType dtype_{nvinfer1::DataType::kFLOAT};
  std::vector<nvinfer1::DataType> idx_dtypes_; // one per index input
  std::vector<int64_t> src_shape_;
  std::vector<int64_t> val_shape_;
  std::vector<std::vector<int64_t>> idx_shapes_; // full shape of each index tensor
  int32_t n_indices_{0}; // number of index tensors

  // Empty field collection — this plugin has no serializable attributes.
  nvinfer1::PluginFieldCollection empty_fc_{0, nullptr};
};

// ---------------------------------------------------------------------------
// ScatterAddPluginCreator
// ---------------------------------------------------------------------------

class ScatterAddPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
  ScatterAddPluginCreator();

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const char* getPluginNamespace() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;

  // Handles both kBUILD and kRUNTIME phases.
  nvinfer1::IPluginV3* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fc,
      nvinfer1::TensorRTPhase phase) noexcept override;

 private:
  nvinfer1::PluginFieldCollection fc_{0, nullptr};
};

} // namespace impl
} // namespace plugins
} // namespace core
} // namespace torch_tensorrt
