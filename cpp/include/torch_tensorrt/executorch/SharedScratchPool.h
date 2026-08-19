/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Bookkeeping for the TensorRT backend's shared per-device activation-scratch
// pool: the grow/reuse/per-device policy and the enqueue-handoff rule.
// Allocation and event creation arrive as callables rather than being made here.

#include <cuda_runtime.h>

#include <cstddef>
#include <unordered_map>
#include <utility>

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

// Claims a device's handoff for a caller about to enqueue against its shared
// scratch, creating the marker's event on first use.
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
SharedScratchHandoff shared_scratch_claim_event(
    std::unordered_map<int, SharedScratchMarker>& markers,
    int device_id,
    CreateEvent create_event) {
  SharedScratchMarker& marker = markers[device_id];
  if (marker.event == nullptr) {
    marker.event = create_event();
  }
  // A slot with no event is never marked, so a failed creation reports nothing to
  // wait for rather than a wait the caller has no event to perform.
  return {marker.event, marker.pending};
}

// The mark precedes the record, so a failed record leaves the slot claiming an
// enqueue the event does not cover -- the caller must then synchronize the stream
// itself before returning the error.
inline cudaEvent_t shared_scratch_mark_in_flight(std::unordered_map<int, SharedScratchMarker>& markers, int device_id) {
  SharedScratchMarker& marker = markers[device_id];
  if (marker.event != nullptr) {
    marker.pending = true;
  }
  return marker.event;
}

// Bookkeeping for a per-device pool of device-memory buffers that grows
// monotonically to the largest requested size.
//
// `alloc` returns nullptr on failure; the slot is then left untouched.
// Allocating before releasing is what makes that true, and it costs peak
// residency: while a slot grows, the old and the new buffer are both resident.
// `release` must leave no in-flight enqueue pointing at the buffer it frees --
// the CUDA caller syncs the device first.
template <typename Alloc, typename Release>
void* shared_scratch_get_or_grow(
    std::unordered_map<int, std::pair<void*, std::size_t>>& pool,
    int device_id,
    std::size_t need,
    std::size_t& out_size,
    Alloc alloc,
    Release release) {
  auto& slot = pool[device_id];
  if (slot.first != nullptr && slot.second >= need) {
    out_size = slot.second;
    return slot.first;
  }
  void* p = alloc(need);
  if (p == nullptr) {
    return nullptr;
  }
  if (slot.first != nullptr) {
    release(slot.first);
  }
  slot = {p, need};
  out_size = need;
  return p;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
