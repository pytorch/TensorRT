/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <executorch/extension/cuda/cuda_allocator.h>
#include <executorch/runtime/core/device_allocator.h>

namespace {

// A program delegated to TensorRT still asks the ExecuTorch runtime for its
// device planned buffers, and the runtime can only serve those once something
// has registered a CUDA DeviceAllocator. The pinned ExecuTorch release does
// that from inside the CUDA/AOTI delegate, which an application that delegates
// to TensorRT alone has no reason to link. Register it here instead.
//
// A C++ consumer may have already loaded the CUDA backend or registered its
// own allocator. Keep that registration: registering the device type twice
// aborts, even when both callers use the same singleton.
[[maybe_unused]] const bool cuda_device_allocator_registered = [] {
  auto& allocator = executorch::backends::cuda::CudaAllocator::instance();
  if (executorch::runtime::get_device_allocator(allocator.device_type()) == nullptr) {
    executorch::runtime::register_device_allocator(&allocator);
  }
  return true;
}();

} // namespace
