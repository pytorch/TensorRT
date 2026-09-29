/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The body of the shared scratch pool's test-only reset.
//
// It lives here rather than in SharedScratchPool.h because that header ships in
// the released source package and this operation frees everything the live pool
// holds without waiting for work in flight against it. This file is carried by a
// testonly Bazel target, is in neither the packaged include tree nor the packaged
// source set, and is compiled by nothing a release build produces. What
// SharedScratchPool leaves reachable is the friend declaration alone, so a
// consumer of the released sources who wants this has to write it.
//
// Safe only with no claim outstanding and no enqueue in flight against a pooled
// buffer; the caller earns that by synchronizing every stream it submitted on and
// joining every thread that called execute(). "No claim outstanding" covers a
// claim whose release is still disposing of a buffer a growth retired: that
// disposal runs with the device's lock dropped and is still reading the marker
// event and the disposal stream, both of which this destroys.

#include "torch_tensorrt/executorch/SharedScratchPool.h"

#include <chrono>
#include <cstddef>
#include <mutex>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

// How long a reset waits, in total, for the device locks. A test calls the reset
// between cases with nothing claimed, so anything held is a leak the run has
// already failed over; the wait only has to outlast a claim still winding down.
//
// Here rather than on SharedScratchPool, which ships in the released source
// package and has nothing that reads this.
inline constexpr std::chrono::seconds kResetLockWait{5};

// Returns every device's slot to the state it had before anything claimed it,
// handing what the slot held to `dispose(device_id, buffer, event, disposal_stream)`
// so the caller can release it. Entries stay in the map, so a reference `get`
// handed out remains valid.
//
// Answers with the number of devices whose lock it could not take within
// kResetLockWait. A lock that never comes free is a leaked claim, and taking the
// locks unconditionally would hang on exactly that defect, in a fixture that
// resets both before and after every case, so the run would end in a target
// timeout with nothing said about the cause. The caller reports it instead.
//
// The deadline covers the whole reset rather than each device, so a leaked lock
// costs the same however many devices the pool has seen. What a leak must not do
// is spend the budget on behalf of the slots after it: the retry below sweeps
// every slot it has not taken yet on each pass, so a slot whose claim is merely
// winding down is tried again inside the window it is free, rather than once at
// whatever instant a leaked neighbour leaves in the budget.
//
// The registry's lock covers one snapshot of which slots exist, and nothing else.
// Every pooled call on every device goes through that lock to find its slot, so a
// reset that held it across the retry below -- which sleeps between passes and
// can run to the whole deadline -- would hold off devices it never touches for
// that long. Dropping it is what the registry's own stability rule licenses:
// entries are never erased and references stay valid, so a slot pointer taken
// under the lock is still good without it. A slot created after the snapshot is
// not reset, which is a caller claiming from a pool this is only safe to reset
// when nothing is claiming from it.
//
// Each slot is then emptied under its own device lock, and what came out of it is
// disposed of afterwards with that dropped as well, because the disposer frees
// device memory and a device-wide free blocks on everything queued on that device
// -- a parked host function included -- which under the device's lock would hold
// off that device's next claimant.
//
// Nothing here waits for an enqueue: the buffer it frees may still be in use by
// one, and the caller is responsible for there being none.
template <typename Dispose>
std::size_t reset_shared_scratch_pool_slots(SharedScratchPool& pool, Dispose dispose) {
  std::vector<std::pair<int, SharedScratchDevice*>> not_taken_yet;
  {
    std::lock_guard<std::mutex> lk(pool.mu_);
    not_taken_yet.reserve(pool.devices_.size());
    for (auto& entry : pool.devices_) {
      not_taken_yet.emplace_back(entry.first, &entry.second);
    }
  }

  std::vector<std::tuple<int, void*, cudaEvent_t, cudaStream_t>> taken;
  taken.reserve(not_taken_yet.size());
  const auto deadline = std::chrono::steady_clock::now() + kResetLockWait;
  while (!not_taken_yet.empty()) {
    std::vector<std::pair<int, SharedScratchDevice*>> still_to_try;
    for (const auto& slot : not_taken_yet) {
      SharedScratchDevice& dev = *slot.second;
      if (!dev.mu.try_lock()) {
        still_to_try.push_back(slot);
        continue;
      }
      std::lock_guard<std::mutex> dev_lk(dev.mu, std::adopt_lock);
      taken.emplace_back(slot.first, dev.buffer, dev.marker.event, dev.disposal_stream);
      dev.buffer = nullptr;
      dev.capacity = 0;
      dev.marker.event = nullptr;
      dev.marker.pending = false;
      dev.disposal_stream = nullptr;
    }
    not_taken_yet.swap(still_to_try);
    if (not_taken_yet.empty() || std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const std::size_t left_alone = not_taken_yet.size();

  for (const auto& slot : taken) {
    dispose(std::get<0>(slot), std::get<1>(slot), std::get<2>(slot), std::get<3>(slot));
  }
  return left_alone;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
