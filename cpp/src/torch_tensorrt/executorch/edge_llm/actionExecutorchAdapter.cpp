/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "edge_llm/actionExecutorchAdapter.h"

#include "common/logger.h"
#include "edge_llm/languageExecutorchAdapter.h"

#include <exception>
#include <memory>
#include <utility>

namespace trt_edgellm {
namespace executorch {

struct ActionExecutorchAdapter::Impl {
  std::unique_ptr<LanguageExecutorchAdapter> step{};
};

ActionExecutorchAdapter::~ActionExecutorchAdapter() noexcept = default;

std::unique_ptr<ActionExecutorchAdapter> ActionExecutorchAdapter::create(
    SerializedEngineView engine,
    std::vector<std::string> input_names,
    std::vector<std::string> output_names,
    cudaStream_t stream) noexcept {
  try {
    auto adapter = std::unique_ptr<ActionExecutorchAdapter>(new ActionExecutorchAdapter());
    adapter->impl_ = std::make_unique<Impl>();
    adapter->impl_->step = LanguageExecutorchAdapter::create(
        engine,
        std::move(input_names),
        std::move(output_names),
        stream,
        /*profile_index=*/0);
    if (!adapter->impl_->step) {
      LOG_ERROR("ActionExecutorchAdapter: failed to create action-step engine");
      return nullptr;
    }
    return adapter;
  } catch (std::exception const& error) {
    LOG_ERROR("ActionExecutorchAdapter: initialization failed: %s", error.what());
    return nullptr;
  }
}

bool ActionExecutorchAdapter::execute(
    std::vector<TensorView> const& inputs,
    std::vector<TensorView> const& outputs,
    cudaStream_t stream) noexcept {
  if (!impl_ || !impl_->step) {
    LOG_ERROR("ActionExecutorchAdapter: adapter is not initialized");
    return false;
  }
  return impl_->step->execute(inputs, outputs, stream);
}

} // namespace executorch
} // namespace trt_edgellm
