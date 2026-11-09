/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The steps of execute()'s pooled path that read what TensorRT reports, split
// out so the backend's test can drive install_pooled_scratch directly. They are
// declared beside the source that defines them rather than in the installed
// header set: they are implementation details of execute(), and a declaration in
// an installed header is API the next release has to keep. SharedScratchPool.h
// sits here for the same reason.

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

// Returns the activation scratch the shapes bound on `ctx`, a kUSER_MANAGED
// context, need. Where TensorRT reports that it cannot work that out, returns
// the upper bound for the context's optimization profile instead, and logs that
// it did, naming `device_id`.
//
// updateDeviceMemorySizeForShapes fails that way, answering zero, for some
// engines with a data-dependent shape inside them, until the context's first
// enqueue. setDeviceMemoryV2's documentation names the profile's figure as an
// upper bound, so the call can run on it, and it is no more than a kSTATIC
// context allocates.
std::size_t pooled_scratch_bytes_for_shapes(nvinfer1::IExecutionContext& ctx, int device_id);

} // namespace executorch_backend
} // namespace torch_tensorrt
