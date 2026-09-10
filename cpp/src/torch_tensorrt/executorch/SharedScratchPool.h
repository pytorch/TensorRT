/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Bookkeeping for the TensorRT backend's shared per-device activation-scratch
// pool: the grow/reuse policy, the enqueue-handoff rule, the stream a growth
// disposes on, and the lock that scopes them to a single device.
// Allocation, event creation and stream creation arrive as callables rather than
// being made here.

#include <cuda_runtime.h>

#include <cstddef>
#include <mutex>
#include <unordered_map>

namespace torch_tensorrt {
namespace executorch_backend {

// Per-device handoff marker for the shared scratch buffer: the pool-owned CUDA
// event that the last enqueue against the buffer was recorded on.
struct SharedScratchMarker {
  // Nothing in the normal path destroys this event; only the test-only reset
  // hands it to a disposer that does.
  cudaEvent_t event = nullptr;
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
// A claimant holds `mu` from the wait on the previous enqueue through its own
// enqueue and the record of that enqueue on the marker. Holding it that far is
// what makes the marker a complete account of who is using the buffer. Anything
// less leaves an enqueue live in a window the marker does not cover, and a
// claimant entering that window is handed the same buffer with nothing ordering
// the two. `mu` covers one device, so a growth holds no lock a claim on
// another device has to acquire.
struct SharedScratchDevice {
  std::mutex mu;
  void* buffer = nullptr;
  std::size_t capacity = 0;
  SharedScratchMarker marker;
  // The stream a growth queues its free of the buffer it replaced on. Created on
  // the first growth that has one to dispose of; nothing in the normal path
  // destroys it, only the test-only reset hands it to a disposer that does.
  cudaStream_t disposal_stream = nullptr;
};

class SharedScratchPool;

// Defined in SharedScratchPoolReset.h; see the friend declaration inside
// SharedScratchPool for why the body is not here.
template <typename Dispose>
std::size_t reset_shared_scratch_pool_slots(SharedScratchPool& pool, Dispose dispose);

// Holds one SharedScratchDevice per device id.
//
// `get` locks only long enough to find or create the entry, and the reference it
// returns stays usable once that lock is dropped: std::unordered_map keeps
// references to elements valid across rehashing, and entries are never erased.
// This one lock is shared by every device, so holding it couples every device to
// whoever holds it: it covers the lookup, and in the test-only reset a snapshot
// of which slots exist, and nothing else. Nothing is made under it that waits --
// no CUDA call, and no wait for a device's own lock.
class SharedScratchPool {
 public:
  SharedScratchDevice& get(int device_id) {
    std::lock_guard<std::mutex> lk(mu_);
    return devices_[device_id];
  }

 private:
  // The reset is test-only and its body is not here: it frees everything the live
  // pool holds without waiting for work in flight against it, which is not a thing
  // this header should hand a consumer of the released source package -- and this
  // header does ship with those sources. The body is in SharedScratchPoolReset.h,
  // which is testonly and ships nowhere, so a release build can reach the grant and
  // nothing else, and reaching the pool that way means writing the traversal, the
  // locks and the deadline again.
  //
  // It is an open door and not a lock. The grant is on a template, so it admits any
  // definition of that name a consumer writes in this namespace, and such a
  // definition reaches both members below -- including erasing entries a live `get`
  // reference points into, which the stability rule above forbids. C++ offers
  // nothing narrower: a non-template friend admits a definition of its own just the
  // same.
  template <typename Dispose>
  friend std::size_t reset_shared_scratch_pool_slots(SharedScratchPool& pool, Dispose dispose);

  std::mutex mu_;
  std::unordered_map<int, SharedScratchDevice> devices_;
};

// The process-wide per-device pool the TensorRT backend runs on. One buffer per
// device, grown to the largest requirement any call on that device has asked
// for, serves every kUSER_MANAGED context on it that needs activation scratch --
// a context whose engine needs none under any shape claims nothing here -- instead
// of each of N layer-engines pinning its own scratch, which makes device memory
// scale with the layer count and OOMs multi-layer models.
//
// ORDERING: a context reads and writes its scratch for the whole enqueue, which
// can still be in flight when execute() returns, so two enqueues must never hold
// one buffer at the same time. A device's lock is what enforces that -- see the
// backend's SharedScratchClaim -- and it is held from the claim through the
// enqueue and the record of it, so two execute() calls on one device are
// serialized at submission. The lock does not couple two devices: each carries
// its own, and no CUDA call is made under the one lock the registry itself holds.
//
// The buffers, the events and the disposal streams are intentionally never
// released at teardown, and the C++ object is never destroyed either. Nothing here
// runs a CUDA call at process exit, which keeps the pool clear of teardown-order
// hazards against anything else holding device memory; and static destruction
// would destroy the registry's mutex, every device's mutex and the map nodes a
// live reference points into while a thread between the lookup and the release of
// its device lock still holds them. Leaking it costs one allocation.
//
// Inline, so the pool's test hooks can reach the same instance from the
// translation unit that defines them without the backend having to export an
// accessor for them; see SharedScratchPoolTestHooks.h. One instance still, for
// the usual reason a function-local static in an inline function is one.
inline SharedScratchPool& scratch_pool() {
  static SharedScratchPool* const pool = new SharedScratchPool();
  return *pool;
}

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

// The stream this device's growths queue their frees on, creating it on first
// use. Call with `dev.mu` held.
//
// `create_stream` returns a CUDA stream, or nullptr if one could not be created,
// in which case the slot stays empty and the next growth retries.
//
// It is the pool's own stream and not the claimant's: a cudaFreeAsync of a
// cudaMalloc'd pointer, which is what this pool holds, hands the bytes back only at
// the next synchronize of the stream it was queued on, and only the pool can
// promise that synchronize -- a claimant on the path this pool exists for never
// synchronizes its stream and is free to destroy it the moment the call returns.
// The pool makes that synchronize itself, in the same call that queued the free.
template <typename CreateStream>
cudaStream_t shared_scratch_disposal_stream(SharedScratchDevice& dev, CreateStream create_stream) {
  if (dev.disposal_stream == nullptr) {
    dev.disposal_stream = create_stream();
  }
  return dev.disposal_stream;
}

// A buffer a growth replaced, handed back for the caller to dispose of.
//
// A non-null `wait_for` is the marker's event, on which an enqueue that may still
// be reading and writing `buffer` has been recorded; the caller must not free the
// buffer ahead of that enqueue. A null `wait_for` means nothing was ever recorded
// against it, so nothing is using it.
//
// One event covers every enqueue the buffer ever served, but only because each of
// them claims the handoff before enqueueing -- which orders its stream after the
// event -- and records on the event afterwards, so the latest recording completes
// only once all the earlier ones have. An enqueue that reaches the buffer without
// doing both is ordered against nothing here.
//
// The disposal belongs outside `dev.mu`: it waits on the host for that enqueue,
// and under the lock that wait is one an unrelated claim on this device would sit
// through for nothing.
struct RetiredScratch {
  void* buffer = nullptr;
  cudaEvent_t wait_for = nullptr;
};

// Bookkeeping for a device's scratch buffer, which grows monotonically to the
// largest requested size. Call with `dev.mu` held.
//
// `alloc` returns nullptr on failure; the buffer is then left untouched.
// Allocating before releasing is what makes that true, and it costs peak
// residency: while the buffer grows, the old and the new one are both resident.
//
// A growth reports the buffer it displaced through `out_retired`; see
// RetiredScratch for what the caller owes it. Nothing is freed here, so a caller
// that ignores `out_retired` leaks rather than frees a buffer an enqueue may
// still be using. Every path clears `out_retired` first, so a caller reusing one
// across calls is not handed a buffer an earlier call already disposed of.
//
// What the buffer ended up sized at is `dev.capacity`, which the caller holds the
// lock for anyway. It is not reported separately, because a caller reading it
// would be reading what an earlier call grew the pool to and not its own request:
// the only figure that belongs on a caller's execution context is the one it
// asked for here.
template <typename Alloc>
void* shared_scratch_get_or_grow(SharedScratchDevice& dev, std::size_t need, Alloc alloc, RetiredScratch& out_retired) {
  out_retired = RetiredScratch{};
  if (dev.buffer != nullptr && dev.capacity >= need) {
    return dev.buffer;
  }
  void* p = alloc(need);
  if (p == nullptr) {
    return nullptr;
  }
  if (dev.buffer != nullptr) {
    out_retired.buffer = dev.buffer;
    out_retired.wait_for = dev.marker.pending ? dev.marker.event : nullptr;
  }
  dev.buffer = p;
  dev.capacity = need;
  return p;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
