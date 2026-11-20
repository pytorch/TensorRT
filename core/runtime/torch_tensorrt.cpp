/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_tensorrt/torch_tensorrt.h"

#include <cuda_runtime.h>
#include <iostream>
#include "core/util/build_info.h"
#include "core/util/macros.h"

namespace torch_tensorrt {

std::string get_build_info() {
  return std::string("Torch-TensorRT Version: ") + TORCH_TENSORRT_VERSION + '\n' + core::util::get_build_info();
}

void dump_build_info() {
  std::cout << get_build_info() << std::endl;
}

void set_device(int gpu_id) {
  TORCHTRT_ASSERT(cudaSetDevice(gpu_id) == cudaSuccess, "Unable to set CUDA device: " << gpu_id);
}

} // namespace torch_tensorrt
