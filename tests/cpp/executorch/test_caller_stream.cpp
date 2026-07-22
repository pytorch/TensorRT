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

TEST(CallerStreamTest, NestedGuardsRestoreOuterSelection) {
  namespace cuda = executorch::extension::cuda;
  using torch_tensorrt::executorch_backend::getTensorRTExecutionStream;

  const cudaStream_t outer = fake_stream(0);
  const cudaStream_t inner = fake_stream(1);

  EXPECT_EQ(getTensorRTExecutionStream(), cudaStreamPerThread);
  {
    cuda::CallerStreamGuard outer_guard(outer);
    EXPECT_EQ(getTensorRTExecutionStream(), outer);
    {
      cuda::CallerStreamGuard inner_guard(inner);
      EXPECT_EQ(getTensorRTExecutionStream(), inner);
      {
        // An explicit null nested inside a non-null guard must be honored, then
        // restore the enclosing selection on scope exit.
        cuda::CallerStreamGuard null_guard(nullptr);
        EXPECT_EQ(getTensorRTExecutionStream(), nullptr);
      }
      EXPECT_EQ(getTensorRTExecutionStream(), inner);
    }
    EXPECT_EQ(getTensorRTExecutionStream(), outer);
  }
  EXPECT_EQ(getTensorRTExecutionStream(), cudaStreamPerThread);
}

} // namespace
