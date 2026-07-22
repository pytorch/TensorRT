/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <gtest/gtest.h>
#include <torch_tensorrt/executorch/TensorRTCallerStream.h>

namespace {

cudaStream_t fake_stream(int index) {
  static char storage[2];
  return reinterpret_cast<cudaStream_t>(&storage[index]);
}

TEST(CallerStreamTest, GuardSelectsTensorRTExecutionStream) {
  EXPECT_EQ(torch_tensorrt::executorch_backend::getTensorRTExecutionStream(), cudaStreamPerThread);

  const cudaStream_t stream = fake_stream(0);
  {
    executorch::extension::cuda::CallerStreamGuard guard(stream);
    EXPECT_EQ(torch_tensorrt::executorch_backend::getTensorRTExecutionStream(), stream);
  }

  EXPECT_EQ(torch_tensorrt::executorch_backend::getTensorRTExecutionStream(), cudaStreamPerThread);
}

TEST(CallerStreamTest, ExplicitNullStreamRemainsSelected) {
  executorch::extension::cuda::CallerStreamGuard guard(nullptr);
  EXPECT_EQ(torch_tensorrt::executorch_backend::getTensorRTExecutionStream(), nullptr);
}

} // namespace
