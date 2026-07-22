/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_tensorrt/executorch/TensorRTCallerStream.h"

#include <executorch/extension/cuda/caller_stream.h>

namespace torch_tensorrt::executorch_backend {

cudaStream_t getTensorRTExecutionStream() {
  return executorch::extension::cuda::getCallerStream().value_or(cudaStreamPerThread);
}

} // namespace torch_tensorrt::executorch_backend
