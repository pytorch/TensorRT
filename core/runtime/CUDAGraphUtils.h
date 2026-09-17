/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <ATen/cuda/CUDAGraph.h>

#include <utility>

namespace torch_tensorrt {
namespace core {
namespace runtime {

template <typename CaptureFn>
void capture_cudagraph_safely(at::cuda::CUDAGraph& cudagraph, CaptureFn&& capture_fn) {
  cudagraph.capture_begin();
  bool capture_end_attempted = false;

  try {
    std::forward<CaptureFn>(capture_fn)();
    // cudaStreamEndCapture ends capture even when it reports an error, so do
    // not attempt to end the same capture twice.
    capture_end_attempted = true;
    cudagraph.capture_end();
  } catch (...) {
    if (!capture_end_attempted) {
      try {
        cudagraph.capture_end();
      } catch (...) {
        // Preserve the original capture failure. capture_end() has still
        // removed the stream from capture mode before reporting its error.
      }
    }

    try {
      cudagraph.reset();
    } catch (...) {
      // Preserve the original capture failure.
    }
    throw;
  }
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
