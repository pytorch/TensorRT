/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Views of the shared activation-scratch pool that only a test has any business
// calling: one reads the pool's capacity, the other frees everything it holds
// with no wait for work in flight against it.
//
// They live in this header and SharedScratchPoolTestHooks.cpp rather than beside
// the pool, and neither file is compiled into libexecutorch_trt_backend or
// shipped in the source package, so a released build contains no definition of
// either and exports no symbol for them. The Bazel target that carries them is
// testonly. Reaching the pool from a separate translation unit is what
// scratch_pool() is inline in SharedScratchPool.h for.
//
// Both are safe only with no claim outstanding and no enqueue in flight against a
// pooled buffer; a test earns that by synchronizing every stream it submitted on
// and joining every thread that called execute(). A claim is still outstanding
// while its release disposes of a buffer a growth retired: that disposal runs with
// the device's lock dropped and goes on reading the marker event and the disposal
// stream, both of which the reset destroys. The reset also frees what the pool
// holds, so a test should destroy its delegate handles before calling it.

#include <cstddef>

namespace torch_tensorrt {
namespace executorch_backend {

// The bytes the pool holds for `device_id` right now; zero if it holds nothing.
std::size_t shared_scratch_capacity_for_testing(int device_id);

// Frees every device's buffer, destroys its handoff event and its disposal
// stream, and clears the marker, so one test does not inherit a pool an earlier
// one grew.
//
// False when a device's slot still had its lock held at the end of the pool's
// reset deadline; that slot is left as it was. A caller should report it rather
// than carry on, and must not wait for the lock itself -- a lock held for that
// long is a claim that was never released, so nothing is going to release it.
bool reset_shared_scratch_pool_for_testing();

} // namespace executorch_backend
} // namespace torch_tensorrt
