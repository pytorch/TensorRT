/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "edge_llm/executorchAdapterTypes.h"

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace trt_edgellm {
namespace executorch {

class ActionExecutorchAdapter final : public ExecutorchComponentAdapter {
 public:
  static std::unique_ptr<ActionExecutorchAdapter> create(
      SerializedEngineView engine,
      std::vector<std::string> input_names,
      std::vector<std::string> output_names,
      cudaStream_t stream) noexcept;

  ~ActionExecutorchAdapter() noexcept override;

  bool execute(
      std::vector<TensorView> const& inputs,
      std::vector<TensorView> const& outputs,
      cudaStream_t stream) noexcept override;

 private:
  ActionExecutorchAdapter() = default;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace executorch
} // namespace trt_edgellm
