/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "core/conversion/tensorcontainer/TensorContainer.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace {

static auto tensor_container =
    torch::class_<TensorContainer>("_torch_tensorrt_eval_ivalue_types", "TensorContainer").def(torch::init<>());
} // namespace
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
