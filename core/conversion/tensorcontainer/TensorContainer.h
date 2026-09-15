/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <memory>

#include "NvInfer.h"
#include "torch/custom_class.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {

struct TensorContainer : torch::CustomClassHolder {
  nvinfer1::ITensor* tensor_;
  TensorContainer() {}

  void hold_tensor(nvinfer1::ITensor* tensor) {
    tensor_ = tensor;
  }

  nvinfer1::ITensor* tensor() {
    return tensor_;
  }
};

} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
