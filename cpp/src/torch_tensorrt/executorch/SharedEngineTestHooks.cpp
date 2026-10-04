/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_tensorrt/executorch/SharedEngineTestHooks.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

namespace torch_tensorrt {
namespace executorch_backend {

nvinfer1::IRuntime* shared_runtime_for_testing() {
  return TensorRTBackend::shared_runtime();
}

} // namespace executorch_backend
} // namespace torch_tensorrt
