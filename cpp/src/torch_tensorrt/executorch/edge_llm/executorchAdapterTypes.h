/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm {
namespace executorch {

struct SerializedEngineView {
  void const* data{nullptr};
  std::size_t size{0};
};

struct TensorView {
  void* data{nullptr};
  std::vector<int64_t> shape;
  nvinfer1::DataType dtype{nvinfer1::DataType::kHALF};
};

class ExecutorchComponentAdapter {
 public:
  virtual ~ExecutorchComponentAdapter() noexcept = default;

  virtual bool execute(
      std::vector<TensorView> const& inputs,
      std::vector<TensorView> const& outputs,
      cudaStream_t stream) noexcept = 0;
};

} // namespace executorch
} // namespace trt_edgellm
