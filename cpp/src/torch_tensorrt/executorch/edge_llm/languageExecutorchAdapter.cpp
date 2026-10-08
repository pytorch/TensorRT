/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_llm/languageExecutorchAdapter.h"

#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"

#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>

namespace trt_edgellm {
namespace executorch {
namespace {

bool to_dims(std::vector<int64_t> const& shape, nvinfer1::Dims& dims) noexcept {
  if (shape.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
    return false;
  }
  dims.nbDims = static_cast<int32_t>(shape.size());
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    int64_t const extent = shape[static_cast<std::size_t>(i)];
    if (extent < 0 || extent > std::numeric_limits<int32_t>::max()) {
      return false;
    }
    dims.d[i] = static_cast<int32_t>(extent);
  }
  return true;
}

bool same_shape(nvinfer1::Dims const& dims, std::vector<int64_t> const& shape) noexcept {
  if (dims.nbDims != static_cast<int32_t>(shape.size())) {
    return false;
  }
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] != shape[static_cast<std::size_t>(i)]) {
      return false;
    }
  }
  return true;
}

} // namespace

struct LanguageExecutorchAdapter::Impl {
  std::unique_ptr<void, DlDeleter> plugin_handle{};
  std::unique_ptr<nvinfer1::IRuntime> runtime{};
  std::unique_ptr<nvinfer1::ICudaEngine> engine{};
  AuxStreamSet aux_streams{};
  std::unique_ptr<nvinfer1::IExecutionContext> context{};
  rt::Tensor context_memory{};
  std::vector<std::string> input_names{};
  std::vector<std::string> output_names{};
};

LanguageExecutorchAdapter::~LanguageExecutorchAdapter() noexcept = default;

std::unique_ptr<LanguageExecutorchAdapter> LanguageExecutorchAdapter::create(
    SerializedEngineView engine_view,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    cudaStream_t stream,
    int32_t profile_index) noexcept {
  try {
    if (engine_view.data == nullptr || engine_view.size == 0 || input_names.empty() || output_names.empty() ||
        profile_index < 0) {
      LOG_ERROR(
          "LanguageExecutorchAdapter: invalid engine, binding names, or "
          "profile index");
      return nullptr;
    }

    auto adapter = std::unique_ptr<LanguageExecutorchAdapter>(new LanguageExecutorchAdapter());
    adapter->impl_ = std::make_unique<Impl>();
    auto& impl = *adapter->impl_;
    impl.plugin_handle = loadEdgellmPluginLib();
    if (!impl.plugin_handle) {
      LOG_ERROR("LanguageExecutorchAdapter: failed to load Edge-LLM plugins");
      return nullptr;
    }

    impl.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!impl.runtime) {
      LOG_ERROR("LanguageExecutorchAdapter: failed to create TensorRT runtime");
      return nullptr;
    }
    impl.engine =
        std::unique_ptr<nvinfer1::ICudaEngine>(impl.runtime->deserializeCudaEngine(engine_view.data, engine_view.size));
    if (!impl.engine) {
      LOG_ERROR("LanguageExecutorchAdapter: failed to deserialize language engine");
      return nullptr;
    }

    for (int32_t index = 0; index < impl.engine->getNbIOTensors(); ++index) {
      char const* name = impl.engine->getIOTensorName(index);
      if (impl.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        impl.input_names.emplace_back(name);
      } else {
        impl.output_names.emplace_back(name);
      }
    }
    if (impl.input_names.size() != input_names.size() || impl.output_names.size() != output_names.size()) {
      LOG_ERROR(
          "LanguageExecutorchAdapter: serialized binding counts do not match "
          "the engine");
      return nullptr;
    }

    if (profile_index >= impl.engine->getNbOptimizationProfiles()) {
      LOG_ERROR("LanguageExecutorchAdapter: profile %d is not present in the engine", profile_index);
      return nullptr;
    }
    impl.context = std::unique_ptr<nvinfer1::IExecutionContext>(
        impl.engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    if (!impl.context || !impl.context->setOptimizationProfileAsync(profile_index, stream)) {
      LOG_ERROR(
          "LanguageExecutorchAdapter: failed to create execution context for "
          "profile %d",
          profile_index);
      return nullptr;
    }
    setNonBlockingAuxStreams(impl.context.get(), impl.engine.get(), impl.aux_streams);

    int64_t const context_bytes = impl.engine->getDeviceMemorySizeV2();
    if (context_bytes > 0) {
      impl.context_memory = rt::Tensor(
          {context_bytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "LanguageExecutorchAdapter::context");
      impl.context->setDeviceMemoryV2(impl.context_memory.rawPointer(), impl.context_memory.getMemoryCapacity());
    }
    return adapter;
  } catch (std::exception const& error) {
    LOG_ERROR("LanguageExecutorchAdapter: initialization failed: %s", error.what());
    return nullptr;
  }
}

bool LanguageExecutorchAdapter::execute(
    std::vector<TensorView> const& inputs,
    std::vector<TensorView> const& outputs,
    cudaStream_t stream) noexcept {
  try {
    if (!impl_ || inputs.size() != impl_->input_names.size() || outputs.size() != impl_->output_names.size()) {
      LOG_ERROR(
          "LanguageExecutorchAdapter: argument count does not match engine "
          "bindings");
      return false;
    }

    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto const& view = inputs[i];
      auto const& name = impl_->input_names[i];
      nvinfer1::Dims dims{};
      if (view.data == nullptr || view.dtype != impl_->engine->getTensorDataType(name.c_str()) ||
          !to_dims(view.shape, dims) || !impl_->context->setInputShape(name.c_str(), dims) ||
          !impl_->context->setTensorAddress(name.c_str(), view.data)) {
        LOG_ERROR("LanguageExecutorchAdapter: failed to bind input %s", name.c_str());
        return false;
      }
    }

    if (!impl_->context->allInputDimensionsSpecified()) {
      LOG_ERROR(
          "LanguageExecutorchAdapter: not all dynamic input dimensions were "
          "specified");
      return false;
    }

    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto const& view = outputs[i];
      auto const& name = impl_->output_names[i];
      nvinfer1::Dims const resolved_shape = impl_->context->getTensorShape(name.c_str());
      if (view.data == nullptr || view.dtype != impl_->engine->getTensorDataType(name.c_str()) ||
          !same_shape(resolved_shape, view.shape) || !impl_->context->setTensorAddress(name.c_str(), view.data)) {
        LOG_ERROR("LanguageExecutorchAdapter: failed to bind output %s", name.c_str());
        return false;
      }
    }

    if (!impl_->context->enqueueV3(stream)) {
      LOG_ERROR("LanguageExecutorchAdapter: TensorRT enqueue failed");
      return false;
    }
    return true;
  } catch (std::exception const& error) {
    LOG_ERROR("LanguageExecutorchAdapter: execute failed: %s", error.what());
    return false;
  }
}

} // namespace executorch
} // namespace trt_edgellm
