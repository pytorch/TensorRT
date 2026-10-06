/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <NvInfer.h>

namespace torch_tensorrt {
namespace executorch_backend {

// For allocator fault injection; change the allocator only while no loads are running.
nvinfer1::IRuntime* shared_runtime_for_testing();

} // namespace executorch_backend
} // namespace torch_tensorrt
