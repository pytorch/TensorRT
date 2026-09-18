/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A second translation unit, so the pool's test target can see whether
// scratch_pool() hands the same registry to every one of them.
//
// It has to be a separate file: the property is a linkage one, and a single-file
// target cannot observe it. scratch_pool() is inline for exactly this reason --
// the backend and the pool's test hooks are different translation units and both
// have to reach the one registry -- and with internal linkage instead each would
// get a registry of its own, the reset hook would clear one nobody uses and the
// capacity hook would always answer zero.
//
// The backend suite does not let that through: six of its assertions across five
// cases read the capacity back and expect a figure above zero, and every one of
// them would fail. What it cannot do is say why, since those cases are named for
// the empty input, the capture refusal and the three growths rather than for
// linkage. That is what this file buys -- one named case in the pool's own target,
// which needs no GPU.

#include "torch_tensorrt/executorch/SharedScratchPool.h"

namespace torch_tensorrt {
namespace executorch_backend {

const SharedScratchPool* scratch_pool_seen_from_another_translation_unit() {
  return &scratch_pool();
}

} // namespace executorch_backend
} // namespace torch_tensorrt
