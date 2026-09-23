/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The one step of execute()'s pooled path that the backend's test drives
// directly, declared beside the source that defines it rather than in the
// installed header set: it is an implementation detail of execute(), and a
// declaration in an installed header is API the next release has to keep.
// SharedScratchPool.h sits here for the same reason.

#include <NvInfer.h>

#include <cstddef>

namespace torch_tensorrt {
namespace executorch_backend {

// Installs `bytes` of `buffer` as the activation scratch of `ctx`, a
// kUSER_MANAGED context, and reports whether TensorRT accepted it. Logs the
// refusal, naming `device_id`, when it did not.
//
// The check is the point. setDeviceMemoryV2 returns void and refuses a buffer
// smaller than the bound shapes need, so a caller that does not ask cannot tell an
// accepted install from a refused one -- and a refused one leaves the context
// pointed at the buffer it was last given, which a shared-scratch pool growth may
// since have freed. The engine then reads and writes freed memory with enqueueV3
// reporting success. The refusal is read back through an IErrorRecorder scoped to
// this one call, the only channel that hands it to the caller: with no recorder
// attached TensorRT writes it to the runtime's ILogger and this function has
// nothing to return.
//
// The backend's own test calls it over the same TensorRT call to cover a refusal
// that cannot be induced through execute().
bool install_pooled_scratch(nvinfer1::IExecutionContext& ctx, void* buffer, std::size_t bytes, int device_id);

} // namespace executorch_backend
} // namespace torch_tensorrt
