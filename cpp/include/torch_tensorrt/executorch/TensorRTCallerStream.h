/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>

namespace torch_tensorrt::executorch_backend {

// Returns the caller-selected stream, or cudaStreamPerThread when no caller
// stream is active.
cudaStream_t getTensorRTExecutionStream();

} // namespace torch_tensorrt::executorch_backend
