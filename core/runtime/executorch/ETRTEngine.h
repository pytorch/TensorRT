/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Slim, libtorch-free engine handle for the ExecuTorch backend.
 *
 * This struct owns only the TRT objects and parsed metadata that the
 * ExecuTorch backend (TensorRTBackend.cpp) actually touches. It is a
 * deliberate, parallel-but-minimal alternative to
 *   core/runtime/TRTEngine.{h,cpp}
 * which pulls in libtorch / c10 / ATen via core/util/prelude.h and friends.
 *
 * Keeping ETRTEngine self-contained lets the ExecuTorch backend
 * link cleanly into binaries that do NOT also link libtorch (e.g.
 * portable executor_runner). The JIT/AOTI runtime, which legitimately
 * depends on libtorch, continues to use core::runtime::TRTEngine.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace torch_tensorrt {
namespace executorch_backend {

struct TrtDeleter {
  template <class T>
  void operator()(T* p) const {
    if (p) {
      delete p;
    }
  }
};

struct ETRTEngine {
  // TRT objects — declaration order matters: destruction is reverse order.
  // exec_ctx must be destroyed before cuda_engine before rt
  // (TensorRT lifetime contract). Do not reorder these three members.
  std::shared_ptr<nvinfer1::IRuntime> rt; // destroyed last
  std::shared_ptr<nvinfer1::ICudaEngine> cuda_engine; // destroyed after exec_ctx
  std::shared_ptr<nvinfer1::IExecutionContext> exec_ctx; // destroyed first

  std::pair<uint64_t, uint64_t> num_io{0, 0};
  std::string name;
  int device_id{0};
  std::vector<std::string> in_binding_names;
  std::vector<std::string> out_binding_names;
  cudaStream_t stream{nullptr};
  std::mutex mu;

  explicit ETRTEngine(std::vector<std::string> serialized_info);
  ~ETRTEngine();

  ETRTEngine(const ETRTEngine&) = delete;
  ETRTEngine& operator=(const ETRTEngine&) = delete;
  ETRTEngine(ETRTEngine&&) = delete;
  ETRTEngine& operator=(ETRTEngine&&) = delete;

  // Wire-format validation. Throws std::runtime_error on mismatch.
  static void verify_serialization_fmt(const std::vector<std::string>& info);
};

} // namespace executorch_backend
} // namespace torch_tensorrt
