/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <stdexcept>

#include "core/runtime/CUDAGraphUtils.h"

TEST(Runtime, CUDAGraphCaptureFailureCleansUpStream) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  auto stream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard stream_guard(stream);
  at::cuda::CUDAGraph cudagraph(/*keep_graph=*/true);

  EXPECT_THROW(
      torch_tensorrt::core::runtime::capture_cudagraph_safely(
          cudagraph, []() { throw std::runtime_error("simulated enqueueV3 failure"); }),
      std::runtime_error);

  cudaStreamCaptureStatus capture_status;
  ASSERT_EQ(cudaStreamIsCapturing(stream.stream(), &capture_status), cudaSuccess);
  EXPECT_EQ(capture_status, cudaStreamCaptureStatusNone);

  // A normal launch and synchronization must still work on the same stream.
  auto output = torch::ones({4}, torch::TensorOptions().device(torch::kCUDA)) + 1;
  ASSERT_EQ(cudaStreamSynchronize(stream.stream()), cudaSuccess);
  EXPECT_TRUE(torch::all(output.cpu() == 2).item<bool>());
}
