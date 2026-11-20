/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <string>
#include "torch_tensorrt/logging.h"
#include "torch_tensorrt/macros.h"

namespace torch_tensorrt {

/** Get the Torch-TensorRT, TensorRT, and LibTorch build information. */
TORCHTRT_API std::string get_build_info();

/** Print the build information to stdout. */
TORCHTRT_API void dump_build_info();

/** Select the active CUDA device for subsequent runtime operations. */
TORCHTRT_API void set_device(int gpu_id);

} // namespace torch_tensorrt
