/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// What makes two loads of the TensorRT backend one shared engine. A header of its own so a test
// can check the key without a GPU.

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>

namespace torch_tensorrt {
namespace executorch_backend {

// The engine bytes, the device, and the weight streaming budget asked for (-1 for TensorRT's
// automatic one), since the budget is fixed before the first context. The bytes are known by their
// hash and size because init frees them; keeping a copy to compare would cost the memory that
// sharing saves. std::hash is not collision resistant, so this trusts the program, as the README
// already asks.
using SharedEngineKey = std::tuple<std::size_t, std::uint64_t, int, std::int64_t>;

inline SharedEngineKey shared_engine_key(const void* data, std::uint64_t size, int device_id, std::int64_t budget) {
  return SharedEngineKey{
      std::hash<std::string_view>{}(std::string_view(static_cast<const char*>(data), size)), size, device_id, budget};
}

} // namespace executorch_backend
} // namespace torch_tensorrt
