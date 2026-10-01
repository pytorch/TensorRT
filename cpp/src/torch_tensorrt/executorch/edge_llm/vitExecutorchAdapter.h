/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "edge_llm/executorchAdapterTypes.h"

#include <cuda_runtime.h>

#include <memory>

namespace trt_edgellm {
namespace executorch {

class VitExecutorchAdapter final : public ExecutorchComponentAdapter {
 public:
  static std::unique_ptr<VitExecutorchAdapter> create(SerializedEngineView engine, cudaStream_t stream) noexcept;

  ~VitExecutorchAdapter() noexcept override;

  bool execute(
      std::vector<TensorView> const& inputs,
      std::vector<TensorView> const& outputs,
      cudaStream_t stream) noexcept override;

 private:
  VitExecutorchAdapter() = default;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace executorch
} // namespace trt_edgellm
