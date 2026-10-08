/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_llm/vitExecutorchAdapter.h"

#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "multimodal/vitRunner.h"

#include <exception>
#include <memory>

namespace trt_edgellm {
namespace executorch {

struct VitExecutorchAdapter::Impl {
  std::unique_ptr<void, DlDeleter> plugin_handle{};
  std::unique_ptr<rt::VitRunner> runner{};
  rt::Tensor context_memory{};
};

VitExecutorchAdapter::~VitExecutorchAdapter() noexcept = default;

std::unique_ptr<VitExecutorchAdapter> VitExecutorchAdapter::create(
    SerializedEngineView engine,
    cudaStream_t stream) noexcept {
  try {
    auto adapter = std::unique_ptr<VitExecutorchAdapter>(new VitExecutorchAdapter());
    adapter->impl_ = std::make_unique<Impl>();
    adapter->impl_->plugin_handle = loadEdgellmPluginLib();
    if (!adapter->impl_->plugin_handle) {
      LOG_ERROR("VitExecutorchAdapter: failed to load Edge-LLM plugins");
      return nullptr;
    }

    adapter->impl_->runner =
        std::make_unique<rt::VitRunner>(rt::SerializedEngineView{engine.data, engine.size}, stream);
    int64_t const context_bytes = adapter->impl_->runner->getRequiredContextMemorySize();
    if (context_bytes > 0) {
      adapter->impl_->context_memory =
          rt::Tensor({context_bytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT8, "VitExecutorchAdapter::context");
      if (!adapter->impl_->runner->setContextMemory(adapter->impl_->context_memory)) {
        LOG_ERROR("VitExecutorchAdapter: failed to set TensorRT context memory");
        return nullptr;
      }
    }
    return adapter;
  } catch (std::exception const& error) {
    LOG_ERROR("VitExecutorchAdapter: initialization failed: %s", error.what());
    return nullptr;
  }
}

bool VitExecutorchAdapter::execute(
    std::vector<TensorView> const& inputs,
    std::vector<TensorView> const& outputs,
    cudaStream_t stream) noexcept {
  try {
    if (inputs.size() != 1 || outputs.size() != 1) {
      LOG_ERROR("VitExecutorchAdapter: expected exactly one input and one output");
      return false;
    }
    TensorView const& pixel_values = inputs[0];
    TensorView const& visual_embeds = outputs[0];
    if (pixel_values.data == nullptr || visual_embeds.data == nullptr) {
      LOG_ERROR("VitExecutorchAdapter: input and output pointers must not be null");
      return false;
    }
    rt::Tensor input(
        pixel_values.data,
        rt::Coords(pixel_values.shape),
        rt::DeviceType::kGPU,
        pixel_values.dtype,
        "VitExecutorchAdapter::pixelValues");
    rt::Tensor output(
        visual_embeds.data,
        rt::Coords(visual_embeds.shape),
        rt::DeviceType::kGPU,
        visual_embeds.dtype,
        "VitExecutorchAdapter::visualEmbeds");
    return impl_->runner->executePrepared(input, output, stream);
  } catch (std::exception const& error) {
    LOG_ERROR("VitExecutorchAdapter: execute failed: %s", error.what());
    return false;
  }
}

} // namespace executorch
} // namespace trt_edgellm
