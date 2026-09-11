/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_tensorrt/executorch/SharedScratchPoolTestHooks.h"

#include "torch_tensorrt/executorch/SharedScratchPool.h"
#include "torch_tensorrt/executorch/SharedScratchPoolReset.h"

#include <cuda_runtime.h>

#include <mutex>

namespace torch_tensorrt {
namespace executorch_backend {

std::size_t shared_scratch_capacity_for_testing(int device_id) {
  SharedScratchDevice& dev = scratch_pool().get(device_id);
  std::lock_guard<std::mutex> lk(dev.mu);
  return dev.capacity;
}

bool reset_shared_scratch_pool_for_testing() {
  int restore_to = 0;
  const bool have_current = cudaGetDevice(&restore_to) == cudaSuccess;
  const std::size_t left_alone = reset_shared_scratch_pool_slots(
      scratch_pool(), [](int device_id, void* buffer, cudaEvent_t event, cudaStream_t disposal_stream) {
        if (buffer == nullptr && event == nullptr && disposal_stream == nullptr) {
          return;
        }
        // cudaFree, cudaEventDestroy and cudaStreamDestroy all act on the current
        // device, and a slot is keyed by the device its buffer came from.
        if (cudaSetDevice(device_id) != cudaSuccess) {
          return;
        }
        if (buffer != nullptr) {
          (void)cudaFree(buffer);
        }
        if (event != nullptr) {
          (void)cudaEventDestroy(event);
        }
        if (disposal_stream != nullptr) {
          (void)cudaStreamDestroy(disposal_stream);
        }
        // Clears a non-sticky error so a reset does not leave one for the next call to
        // report. A sticky one survives the clear, and no cleanup here recovers it.
        (void)cudaGetLastError();
      });
  if (have_current) {
    (void)cudaSetDevice(restore_to);
  }
  return left_alone == 0;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
