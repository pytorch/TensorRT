/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Bookkeeping for the TensorRT backend's shared per-device activation-scratch
// pool: the grow/reuse policy, the enqueue-handoff rule, and the lock that scopes
// both to a single device.
// Allocation and event creation arrive as callables rather than being made here.

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace torch_tensorrt {
namespace executorch_backend {

// Runtime backend option that backs execution-context activation scratch with a
// shared per-device pool instead of giving every context its own. Boolean,
// default false. Delivered as
//   executorch::runtime::set_option("TensorRTBackend", options.view())
// A context's allocation strategy is fixed when the context is created, so a
// later call governs only the engines loaded after it, and a pooled context and
// a private-scratch one coexist in one process.
inline constexpr char kSharedActivationScratchKey[] = "use_shared_activation_scratch";

// Per-device handoff marker for the shared scratch buffer: the pool-owned CUDA
// event that the last enqueue against the buffer was recorded on.
struct SharedScratchMarker {
  cudaEvent_t event = nullptr; // never destroyed; one event serves the slot for the process lifetime
  bool pending = false; // an enqueue against the buffer has been recorded on `event`
};

// What a caller about to enqueue against a device's shared scratch has to do:
// when `needs_wait`, make its stream wait on `event` first; once the enqueue is
// submitted, record it on `event`. `event` is null only when the slot has no
// event and one could not be created.
struct SharedScratchHandoff {
  cudaEvent_t event = nullptr;
  bool needs_wait = false;
};

// One device's shared scratch buffer and the marker ordering its handoff, behind
// the lock that covers both.
//
// A claimant holds `mu` from the wait on the previous enqueue through the choice
// of buffer, so it cannot be handed a buffer another claimant is midway through
// replacing, and cannot record its own enqueue against a marker that has since
// moved on. `mu` covers one device, so a growth holds no lock a claim on another
// device has to acquire.
struct SharedScratchDevice {
  std::mutex mu;
  void* buffer = nullptr;
  std::size_t capacity = 0;
  SharedScratchMarker marker;
};

// Holds one SharedScratchDevice per device id.
//
// `get` locks only long enough to find or create the entry, and the reference it
// returns stays usable once that lock is dropped: std::unordered_map keeps
// references to elements valid across rehashing, and entries are never erased.
// This one lock is shared by every device, which is why nothing but the lookup
// runs under it.
class SharedScratchPool {
 public:
  SharedScratchDevice& get(int device_id) {
    std::lock_guard<std::mutex> lk(mu_);
    return devices_[device_id];
  }

 private:
  std::mutex mu_;
  std::unordered_map<int, SharedScratchDevice> devices_;
};

// Claims a device's handoff for a caller about to enqueue against its shared
// scratch, creating the marker's event on first use. Call with `dev.mu` held.
//
// `create_event` returns a CUDA event, or nullptr if one could not be created,
// in which case the slot stays empty and the next call retries.
//
// The ordering between one enqueue and the next is carried by an event rather
// than by the stream the previous enqueue used, because a stream handle cannot
// carry it: synchronizing on a handle whose stream the caller has since
// destroyed is a crash rather than an error return, CUDA recycles handle values
// so a genuinely different stream can compare equal to the recorded one, and the
// NULL stream is both a legal stream a caller can select and the only available
// "no previous user" sentinel. An event names the work instead of the queue --
// it stays valid after the stream that recorded it is destroyed, and waiting on
// it from the stream that recorded it is already satisfied, so the common
// single-stream case costs a host call and no device stall.
template <typename CreateEvent>
SharedScratchHandoff shared_scratch_claim_event(SharedScratchDevice& dev, CreateEvent create_event) {
  if (dev.marker.event == nullptr) {
    dev.marker.event = create_event();
  }
  // A slot with no event is never marked, so a failed creation reports nothing to
  // wait for rather than a wait the caller has no event to perform.
  return {dev.marker.event, dev.marker.pending};
}

// Call with `dev.mu` held.
//
// The mark precedes the record, so a failed record leaves the slot claiming an
// enqueue the event does not cover -- the caller must then synchronize the stream
// itself before returning the error.
inline cudaEvent_t shared_scratch_mark_in_flight(SharedScratchDevice& dev) {
  if (dev.marker.event != nullptr) {
    dev.marker.pending = true;
  }
  return dev.marker.event;
}

// Bookkeeping for a device's scratch buffer, which grows monotonically to the
// largest requested size. Call with `dev.mu` held.
//
// `alloc` returns nullptr on failure; the buffer is then left untouched.
// Allocating before releasing is what makes that true, and it costs peak
// residency: while the buffer grows, the old and the new one are both resident.
//
// `release(old, wait_for)` frees `old`. A non-null `wait_for` is the marker's
// event, on which an enqueue that may still be reading and writing `old` has been
// recorded; the release must wait for that event on the host before freeing. One
// event covers every enqueue the buffer ever served, but only because each of
// them claims the handoff before enqueueing -- which orders its stream after the
// event -- and records on the event afterwards, so the latest recording completes
// only once all the earlier ones have. An enqueue that reaches the buffer without
// doing both is covered by no wait here. A null `wait_for` means nothing was ever
// recorded against this buffer, so there is nothing to wait for.
template <typename Alloc, typename Release>
void* shared_scratch_get_or_grow(
    SharedScratchDevice& dev,
    std::size_t need,
    std::size_t& out_size,
    Alloc alloc,
    Release release) {
  if (dev.buffer != nullptr && dev.capacity >= need) {
    out_size = dev.capacity;
    return dev.buffer;
  }
  void* p = alloc(need);
  if (p == nullptr) {
    return nullptr;
  }
  if (dev.buffer != nullptr) {
    release(dev.buffer, dev.marker.pending ? dev.marker.event : nullptr);
  }
  dev.buffer = p;
  dev.capacity = need;
  out_size = need;
  return p;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
