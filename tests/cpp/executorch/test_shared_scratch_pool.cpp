/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Pins the shared scratch pool helper: its grow, reuse and per-device policy, its
// enqueue-handoff rule and the stream a growth disposes on, driven over fakes so
// no CUDA device is needed.
//
// This exercises the helper, not the backend: it does not link the delegate, so
// it cannot catch the delegate calling the helper wrongly or ceasing to call it.
// test_shared_scratch_backend covers that, and needs a GPU to do it.
//
// One case at the end is the exception to "over fakes": the lifetime of the
// process-wide registry scratch_pool() hands out is a property of that object and
// not of any fake, and it is observable only after main returns. It forks to see
// it, and still makes no CUDA call.

#include "torch_tensorrt/executorch/SharedScratchPool.h"
#include "torch_tensorrt/executorch/SharedScratchPoolReset.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#endif

namespace torch_tensorrt {
namespace executorch_backend {

// Defined in test_shared_scratch_pool_other_tu.cpp; see
// EveryTranslationUnitSeesOneProcessPool below.
const SharedScratchPool* scratch_pool_seen_from_another_translation_unit();

namespace {

// Fake device allocator: hands out distinct non-null pointers and records every
// allocation size and every buffer a growth retired, so tests can assert the
// pool's grow/reuse policy and what each retirement has to wait for, without a
// CUDA device.
struct FakeAllocator {
  std::vector<std::size_t> alloc_sizes;
  std::vector<std::pair<void*, cudaEvent_t>> retirements;
  std::uintptr_t next = 0x1000;
  bool fail_next = false;

  void* alloc(std::size_t bytes) {
    if (fail_next) {
      fail_next = false;
      return nullptr;
    }
    alloc_sizes.push_back(bytes);
    void* p = reinterpret_cast<void*>(next);
    next += 0x1000;
    return p;
  }

  void retire(void* p, cudaEvent_t wait_for) {
    retirements.emplace_back(p, wait_for);
  }

  int alloc_count() const {
    return static_cast<int>(alloc_sizes.size());
  }
};

// Stands in for the CUDA event factory: hands out distinct non-null handles and
// counts calls, so a test can tell a slot that reuses its event from one that
// creates a new one every call.
struct FakeEventFactory {
  int created = 0;
  std::uintptr_t next = 0xE000;
  bool fail_next = false;

  cudaEvent_t operator()() {
    if (fail_next) {
      fail_next = false;
      return nullptr;
    }
    ++created;
    cudaEvent_t e = reinterpret_cast<cudaEvent_t>(next);
    next += 0x100;
    return e;
  }
};

// Stands in for the CUDA stream factory, the way FakeEventFactory stands in for
// the event one: distinct non-null handles and a call count, so a test can tell a
// slot that keeps one stream from one that creates a stream per growth.
struct FakeStreamFactory {
  int created = 0;
  std::uintptr_t next = 0x5000;
  bool fail_next = false;

  cudaStream_t operator()() {
    if (fail_next) {
      fail_next = false;
      return nullptr;
    }
    ++created;
    cudaStream_t s = reinterpret_cast<cudaStream_t>(next);
    next += 0x100;
    return s;
  }
};

// Stands in for the backend: passes the allocator through and records whatever
// the call retired, the way execute() hands a retired buffer to its claim.
void* call(SharedScratchDevice& dev, FakeAllocator& a, std::size_t need) {
  RetiredScratch retired;
  void* const p = shared_scratch_get_or_grow(
      dev, need, [&a](std::size_t bytes) { return a.alloc(bytes); }, retired);
  if (retired.buffer != nullptr) {
    a.retire(retired.buffer, retired.wait_for);
  }
  return p;
}

TEST(SharedScratchPool, FirstRequestAllocatesExactSize) {
  SharedScratchDevice dev;
  FakeAllocator a;

  void* p = call(dev, a, /*need=*/1024);

  EXPECT_NE(p, nullptr);
  EXPECT_EQ(dev.capacity, 1024u);
  ASSERT_EQ(a.alloc_count(), 1);
  EXPECT_EQ(a.alloc_sizes[0], 1024u);
  EXPECT_TRUE(a.retirements.empty());
}

TEST(SharedScratchPool, ReusesWhenExistingBufferIsLargeEnough) {
  SharedScratchDevice dev;
  FakeAllocator a;

  void* first = call(dev, a, 4096);
  // A smaller and an equal request must both reuse the same buffer (no realloc).
  void* second = call(dev, a, 1000);
  void* third = call(dev, a, 4096);

  EXPECT_EQ(second, first);
  EXPECT_EQ(third, first);
  // The smaller request did not shrink the pool to what it asked for.
  EXPECT_EQ(dev.capacity, 4096u);
  EXPECT_EQ(a.alloc_count(), 1);
  EXPECT_TRUE(a.retirements.empty());
}

TEST(SharedScratchPool, GrowsMonotonicallyToMaxAndRetiresOldBuffer) {
  SharedScratchDevice dev;
  FakeAllocator a;

  void* small = call(dev, a, 1024);
  void* big = call(dev, a, 8192);

  EXPECT_NE(big, small);
  EXPECT_EQ(dev.capacity, 8192u);
  ASSERT_EQ(a.alloc_count(), 2);
  EXPECT_EQ(a.alloc_sizes[1], 8192u);
  ASSERT_EQ(a.retirements.size(), 1u);
  EXPECT_EQ(a.retirements[0].first, small);

  // A subsequent smaller request reuses the grown buffer -- pool never shrinks.
  void* reuse = call(dev, a, 512);
  EXPECT_EQ(reuse, big);
  EXPECT_EQ(dev.capacity, 8192u);
  EXPECT_EQ(a.alloc_count(), 2);
}

TEST(SharedScratchPool, GrowRetiresTheOldBufferWithTheEventToWaitOn) {
  SharedScratchDevice dev;
  FakeAllocator a;
  FakeEventFactory events;

  void* small = call(dev, a, 1024);
  ASSERT_NE(small, nullptr);

  // An enqueue against `small` has been submitted and recorded, so its
  // retirement has something specific to outlive.
  const SharedScratchHandoff handoff = shared_scratch_claim_event(dev, std::ref(events));
  ASSERT_EQ(shared_scratch_mark_in_flight(dev), handoff.event);

  ASSERT_NE(call(dev, a, 8192), nullptr);

  ASSERT_EQ(a.retirements.size(), 1u);
  EXPECT_EQ(a.retirements[0].first, small);
  // The retirement carries the event that enqueue was recorded on, so the caller
  // has one specific enqueue to wait for, rather than needing a device-wide
  // synchronize to be correct.
  EXPECT_EQ(a.retirements[0].second, handoff.event);
}

TEST(SharedScratchPool, GrowHasNothingToWaitForWhenNoEnqueueWasRecorded) {
  SharedScratchDevice dev;
  FakeAllocator a;
  FakeEventFactory events;

  void* small = call(dev, a, 1024);
  ASSERT_NE(small, nullptr);
  // The slot has an event, but nothing has been recorded on it: claiming the
  // handoff is not the same as enqueueing against the buffer.
  ASSERT_NE(shared_scratch_claim_event(dev, std::ref(events)).event, nullptr);

  ASSERT_NE(call(dev, a, 8192), nullptr);

  ASSERT_EQ(a.retirements.size(), 1u);
  EXPECT_EQ(a.retirements[0].first, small);
  EXPECT_EQ(a.retirements[0].second, nullptr);
}

TEST(SharedScratchPool, AllocationFailureLeavesExistingBufferUntouched) {
  SharedScratchDevice dev;
  FakeAllocator a;

  void* first = call(dev, a, 1024);
  ASSERT_NE(first, nullptr);

  // A growth whose allocation fails must return nullptr and keep the old buffer,
  // so the caller can surface the error without corrupting the pool.
  a.fail_next = true;
  void* failed = call(dev, a, 8192);
  EXPECT_EQ(failed, nullptr);
  EXPECT_TRUE(a.retirements.empty());

  // The device still holds the original buffer and serves it on the next request.
  void* again = call(dev, a, 1024);
  EXPECT_EQ(again, first);
  EXPECT_EQ(dev.capacity, 1024u);
}

TEST(SharedScratchPool, FirstAllocationFailureReturnsNullAndStoresNothing) {
  SharedScratchDevice dev;
  FakeAllocator a;

  a.fail_next = true;
  void* p = call(dev, a, 1024);
  EXPECT_EQ(p, nullptr);
  EXPECT_EQ(dev.buffer, nullptr);
  EXPECT_EQ(dev.capacity, 0u);

  // Nothing stored: a later successful request allocates fresh.
  void* q = call(dev, a, 1024);
  EXPECT_NE(q, nullptr);
  EXPECT_EQ(a.alloc_count(), 1);
}

// The backend passes one RetiredScratch through a call and acts on what it holds,
// so a path that leaves the previous growth's buffer in it hands that buffer to a
// second caller to free.
TEST(SharedScratchPool, EveryPathClearsTheRetirementItReports) {
  SharedScratchDevice dev;
  FakeAllocator a;
  // Deliberately reused across the calls below, which is the state the clearing is
  // for; `call` above gives each of its calls a fresh one.
  RetiredScratch retired;
  const auto request = [&](std::size_t need) {
    return shared_scratch_get_or_grow(
        dev, need, [&a](std::size_t bytes) { return a.alloc(bytes); }, retired);
  };

  void* const first = request(1024);
  ASSERT_NE(first, nullptr);
  ASSERT_EQ(retired.buffer, nullptr);
  ASSERT_NE(request(8192), nullptr);
  ASSERT_EQ(retired.buffer, first) << "the growth did not report the buffer it replaced";

  EXPECT_NE(request(512), nullptr);
  EXPECT_EQ(retired.buffer, nullptr) << "a reuse left the previous growth's buffer in the result, so a caller acting "
                                        "on it frees a buffer that was already handed back once";

  void* const second = dev.buffer;
  ASSERT_NE(request(16384), nullptr);
  ASSERT_EQ(retired.buffer, second);
  a.fail_next = true;
  EXPECT_EQ(request(65536), nullptr);
  EXPECT_EQ(retired.buffer, nullptr) << "a failed allocation left the previous growth's buffer in the result";
}

// ---------------------------------------------------------------------------
// Ordering the shared buffer's handoff from one enqueue to the next.
// ---------------------------------------------------------------------------

TEST(SharedScratchHandoffTest, FirstUseCreatesTheSlotsEventAndWaitsForNothing) {
  SharedScratchDevice dev;
  FakeEventFactory events;

  const SharedScratchHandoff handoff = shared_scratch_claim_event(dev, std::ref(events));

  EXPECT_NE(handoff.event, nullptr);
  EXPECT_FALSE(handoff.needs_wait);
  EXPECT_EQ(events.created, 1);
}

TEST(SharedScratchHandoffTest, EveryUseAfterAnEnqueueWaitsOnTheSameEvent) {
  SharedScratchDevice dev;
  FakeEventFactory events;
  const SharedScratchHandoff first = shared_scratch_claim_event(dev, std::ref(events));
  ASSERT_FALSE(first.needs_wait);

  EXPECT_EQ(shared_scratch_mark_in_flight(dev), first.event);

  // Every later enqueue waits, however many there have been and whichever stream
  // each of them ran on: the marker records that the buffer was handed out, not
  // who it was handed to. Comparing stream handles instead would let a caller
  // through whenever its handle matched the recorded one, including when CUDA has
  // recycled that value for a different stream.
  const SharedScratchHandoff second = shared_scratch_claim_event(dev, std::ref(events));
  EXPECT_TRUE(second.needs_wait);
  EXPECT_EQ(second.event, first.event);

  const SharedScratchHandoff third = shared_scratch_claim_event(dev, std::ref(events));
  EXPECT_TRUE(third.needs_wait);
  EXPECT_EQ(third.event, first.event);

  // One event serves the slot for its whole life, so the wait never targets an
  // event some earlier enqueue was recorded on.
  EXPECT_EQ(events.created, 1);
}

TEST(SharedScratchHandoffTest, KeepsAnIndependentMarkerPerDevice) {
  SharedScratchPool pool;
  FakeEventFactory events;
  SharedScratchDevice& dev0 = pool.get(0);
  SharedScratchDevice& dev1 = pool.get(1);
  const SharedScratchHandoff first = shared_scratch_claim_event(dev0, std::ref(events));
  ASSERT_EQ(shared_scratch_mark_in_flight(dev0), first.event);

  // Device 1 has its own buffer, so device 0's enqueue is nothing for it to wait
  // on, and it gets its own event.
  const SharedScratchHandoff second = shared_scratch_claim_event(dev1, std::ref(events));
  EXPECT_FALSE(second.needs_wait);
  EXPECT_NE(second.event, first.event);
  EXPECT_EQ(events.created, 2);

  // Marking device 1 does not make device 0 stop waiting, or the other way round.
  ASSERT_EQ(shared_scratch_mark_in_flight(dev1), second.event);
  EXPECT_TRUE(shared_scratch_claim_event(dev0, std::ref(events)).needs_wait);
  EXPECT_TRUE(shared_scratch_claim_event(dev1, std::ref(events)).needs_wait);
}

TEST(SharedScratchHandoffTest, EventCreationFailureIsReportedAndRetried) {
  SharedScratchDevice dev;
  FakeEventFactory events;

  events.fail_next = true;
  const SharedScratchHandoff failed = shared_scratch_claim_event(dev, std::ref(events));
  EXPECT_EQ(failed.event, nullptr);
  EXPECT_FALSE(failed.needs_wait);

  // The failure leaves nothing behind, so the next call tries again and succeeds
  // rather than serving an unusable slot for the rest of the process.
  const SharedScratchHandoff retried = shared_scratch_claim_event(dev, std::ref(events));
  EXPECT_NE(retried.event, nullptr);
  EXPECT_FALSE(retried.needs_wait);
  EXPECT_EQ(events.created, 1);
}

TEST(SharedScratchHandoffTest, ASlotWithNoEventIsNotMarked) {
  SharedScratchDevice dev;
  FakeEventFactory events;

  // Nothing can be recorded without an event, so nothing is claimed to have been.
  EXPECT_EQ(shared_scratch_mark_in_flight(dev), nullptr);

  // Otherwise, once an event is finally created for the slot, the next caller
  // would wait on it believing an enqueue had been recorded on it that never was.
  EXPECT_FALSE(shared_scratch_claim_event(dev, std::ref(events)).needs_wait);
}

// ---------------------------------------------------------------------------
// The stream a growth queues its free on.
// ---------------------------------------------------------------------------

TEST(SharedScratchDisposalStream, FirstUseCreatesTheSlotsStreamAndEveryLaterOneReusesIt) {
  SharedScratchDevice dev;
  FakeStreamFactory streams;

  const cudaStream_t first = shared_scratch_disposal_stream(dev, std::ref(streams));
  EXPECT_NE(first, nullptr);
  EXPECT_EQ(streams.created, 1);

  // A stream per growth would be a stream leaked per growth: nothing destroys one
  // in the normal path, by the same argument that leaves the buffers and the
  // events alone at teardown.
  EXPECT_EQ(shared_scratch_disposal_stream(dev, std::ref(streams)), first);
  EXPECT_EQ(shared_scratch_disposal_stream(dev, std::ref(streams)), first);
  EXPECT_EQ(streams.created, 1);
}

TEST(SharedScratchDisposalStream, StreamCreationFailureIsReportedAndRetried) {
  SharedScratchDevice dev;
  FakeStreamFactory streams;

  streams.fail_next = true;
  // Null is what the disposal reads as "no stream to queue the free on", so it
  // falls back to a device-wide free rather than passing this to cudaFreeAsync,
  // where it would name the legacy default stream.
  EXPECT_EQ(shared_scratch_disposal_stream(dev, std::ref(streams)), nullptr);

  // The failure leaves nothing behind, so the next growth tries again rather than
  // taking the fallback for the rest of the process.
  EXPECT_NE(shared_scratch_disposal_stream(dev, std::ref(streams)), nullptr);
  EXPECT_EQ(streams.created, 1);
}

// A stream belongs to the device that was current when it was created, and the
// free queued on it is of that device's memory. One stream shared across devices
// would put every growth's free on whichever device grew first.
TEST(SharedScratchDisposalStream, KeepsAnIndependentStreamPerDevice) {
  SharedScratchPool pool;
  FakeStreamFactory streams;

  const cudaStream_t zero = shared_scratch_disposal_stream(pool.get(0), std::ref(streams));
  const cudaStream_t one = shared_scratch_disposal_stream(pool.get(1), std::ref(streams));

  EXPECT_NE(zero, nullptr);
  EXPECT_NE(one, nullptr);
  EXPECT_NE(zero, one);
  EXPECT_EQ(streams.created, 2);
  EXPECT_EQ(shared_scratch_disposal_stream(pool.get(0), std::ref(streams)), zero);
}

// ---------------------------------------------------------------------------
// The registry that owns one entry per device.
// ---------------------------------------------------------------------------

TEST(SharedScratchPoolRegistry, KeepsAnIndependentBufferPerDevice) {
  SharedScratchPool pool;
  FakeAllocator a;

  void* dev0 = call(pool.get(0), a, 2048);
  void* dev1 = call(pool.get(1), a, 2048);

  EXPECT_NE(dev0, dev1);
  EXPECT_EQ(a.alloc_count(), 2);
  EXPECT_TRUE(a.retirements.empty());

  // Growing device 1 must not touch device 0's buffer.
  void* dev1_big = call(pool.get(1), a, 9000);
  void* dev0_again = call(pool.get(0), a, 2048);
  EXPECT_NE(dev1_big, dev1);
  EXPECT_EQ(dev0_again, dev0);
  ASSERT_EQ(a.retirements.size(), 1u);
  EXPECT_EQ(a.retirements[0].first, dev1);
}

TEST(SharedScratchPoolRegistry, HandsOutOneStableEntryPerDevice) {
  SharedScratchPool pool;

  SharedScratchDevice* const seven = &pool.get(7);
  EXPECT_EQ(&pool.get(7), seven);
  EXPECT_NE(&pool.get(8), seven);

  // Callers keep using an entry after the registry's lock is dropped, and go on
  // using it across their CUDA calls, so adding devices must not move it.
  std::set<SharedScratchDevice*> distinct;
  for (int id = 0; id < 512; ++id) {
    distinct.insert(&pool.get(id));
  }
  EXPECT_EQ(&pool.get(7), seven);
  // Two devices must never land on one entry, or a claimant is handed another
  // device's buffer as its own. A bounded or folded key space is a plausible way
  // to write this registry and an invisible way to break it.
  EXPECT_EQ(distinct.size(), 512u);
}

// scratch_pool() is inline so that the backend and the pool's test hooks, which
// are separate translation units, reach one registry. Nothing else here can see
// that: every case above builds its own SharedScratchPool. Give the accessor
// internal linkage instead and each translation unit gets a registry of its own,
// at which point the reset hook clears one nobody uses, the capacity hook always
// answers zero, and several of the backend suite's pool assertions pass without
// testing anything -- on a GPU host, which is the only place they run.
//
// The second translation unit is test_shared_scratch_pool_other_tu.cpp.
TEST(SharedScratchPoolRegistry, EveryTranslationUnitSeesOneProcessPool) {
  EXPECT_EQ(scratch_pool_seen_from_another_translation_unit(), &scratch_pool())
      << "two translation units got different process pools, so scratch_pool() no longer has external linkage and "
         "the pool's test hooks operate on a registry the backend never uses";
}

TEST(SharedScratchPoolRegistry, ResetHandsBackEverySlotAndLeavesItEmpty) {
  // The backend test fixture runs this between cases, so every case that reads
  // what the pool holds depends on it. The two things it has to get right are
  // handing the caller everything the slot held -- the buffer, the event and the
  // disposal stream, none of which the pool or the disposer can find afterwards --
  // and clearing the slot, so the next claim allocates instead of reusing a
  // pointer that has been freed.
  SharedScratchPool pool;
  FakeAllocator zero;
  FakeAllocator one;
  FakeEventFactory events;
  FakeStreamFactory streams;

  SharedScratchDevice& dev0 = pool.get(0);
  SharedScratchDevice& dev1 = pool.get(1);
  void* const dev0_buffer = call(dev0, zero, 4096);
  void* const dev1_buffer = call(dev1, one, 2048);
  const cudaEvent_t dev0_event = shared_scratch_claim_event(dev0, std::ref(events)).event;
  shared_scratch_mark_in_flight(dev0);
  const cudaStream_t dev0_stream = shared_scratch_disposal_stream(dev0, std::ref(streams));
  // Device 1 is left with a buffer and neither an event nor a disposal stream,
  // which is the state of a slot whose event creation failed and which never grew:
  // the reset has to cope with nulls there.
  ASSERT_NE(dev0_buffer, nullptr);
  ASSERT_NE(dev1_buffer, nullptr);
  ASSERT_NE(dev0_event, nullptr);
  ASSERT_NE(dev0_stream, nullptr);
  ASSERT_EQ(dev1.marker.event, nullptr);
  ASSERT_EQ(dev1.disposal_stream, nullptr);

  struct Disposal {
    int device_id;
    void* buffer;
    cudaEvent_t event;
    cudaStream_t disposal_stream;
  };
  std::vector<Disposal> disposed;
  reset_shared_scratch_pool_slots(
      pool, [&](int device_id, void* buffer, cudaEvent_t event, cudaStream_t disposal_stream) {
        disposed.push_back({device_id, buffer, event, disposal_stream});
      });

  // The registry iterates in unspecified order, so each device is looked up.
  const auto disposal_for = [&disposed](int device_id) -> const Disposal* {
    for (const Disposal& d : disposed) {
      if (d.device_id == device_id) {
        return &d;
      }
    }
    return nullptr;
  };
  ASSERT_EQ(disposed.size(), 2u) << "the reset skipped a device's slot, whose buffer is then never freed";
  ASSERT_NE(disposal_for(0), nullptr);
  ASSERT_NE(disposal_for(1), nullptr);
  EXPECT_EQ(disposal_for(0)->buffer, dev0_buffer);
  EXPECT_EQ(disposal_for(0)->event, dev0_event);
  EXPECT_EQ(disposal_for(0)->disposal_stream, dev0_stream);
  EXPECT_EQ(disposal_for(1)->buffer, dev1_buffer);
  EXPECT_EQ(disposal_for(1)->event, nullptr);
  EXPECT_EQ(disposal_for(1)->disposal_stream, nullptr);

  // The slot the reset cleared is the one a later lookup finds, so the reads below
  // are of the entry the backend would go on using. One address cannot tell a
  // surviving entry from a recycled allocation, so the invariant itself is pinned
  // by ResetLeavesEveryEntryWhereItWas rather than here.
  EXPECT_EQ(&pool.get(0), &dev0);
  EXPECT_EQ(dev0.buffer, nullptr);
  EXPECT_EQ(dev0.capacity, 0u);
  EXPECT_EQ(dev0.marker.event, nullptr);
  EXPECT_FALSE(dev0.marker.pending);
  EXPECT_EQ(dev0.disposal_stream, nullptr);

  // A smaller request than the freed buffer served: reuse would satisfy it from
  // the stale capacity and allocate nothing, so this is what distinguishes a
  // cleared slot from one the reset only emptied of its event.
  void* const fresh = call(dev0, zero, 1024);
  EXPECT_NE(fresh, nullptr);
  EXPECT_EQ(dev0.capacity, 1024u);
  EXPECT_EQ(zero.alloc_count(), 2) << "the slot was not cleared, so the request reused the freed buffer";
  EXPECT_TRUE(zero.retirements.empty()) << "the cleared slot retired a buffer the reset had already handed back";
}

// The mirror of device 1 in the case above: an event and no buffer. That state is
// reachable, because the handoff event is created before the allocation, so a
// claim whose allocation fails leaves one behind. A reset that skipped slots with
// no buffer would leak that event and pass every other case in this file.
TEST(SharedScratchPoolRegistry, ResetHandsBackTheEventOfASlotThatNeverGotABuffer) {
  SharedScratchPool pool;
  FakeAllocator a;
  FakeEventFactory events;

  SharedScratchDevice& dev = pool.get(3);
  const cudaEvent_t event = shared_scratch_claim_event(dev, std::ref(events)).event;
  ASSERT_NE(event, nullptr);
  a.fail_next = true;
  ASSERT_EQ(call(dev, a, 1024), nullptr);
  ASSERT_EQ(dev.buffer, nullptr) << "the allocation did not fail, so this slot is not the state under test";

  std::vector<std::pair<void*, cudaEvent_t>> disposed;
  reset_shared_scratch_pool_slots(pool, [&](int, void* buffer, cudaEvent_t slot_event, cudaStream_t) {
    disposed.emplace_back(buffer, slot_event);
  });

  ASSERT_EQ(disposed.size(), 1u) << "the reset skipped a slot holding an event and no buffer, so nothing ever destroys "
                                    "that event";
  EXPECT_EQ(disposed[0].first, nullptr);
  EXPECT_EQ(disposed[0].second, event);
  EXPECT_EQ(dev.marker.event, nullptr) << "the slot kept the event the reset handed to the disposer";
}

// The registry hands out a reference and drops its lock, so callers go on using
// that reference; the reset has to empty a slot without moving it. Checking one
// address cannot pin that. An erasing reset returns the node to the allocator,
// which hands the very same address back to the next lookup, so the comparison
// still holds -- and every read through the old reference between the two is of
// freed memory. Enough entries and the allocator cannot reproduce them all.
TEST(SharedScratchPoolRegistry, ResetLeavesEveryEntryWhereItWas) {
  constexpr int kDevices = 512;
  SharedScratchPool pool;
  FakeAllocator alloc;
  FakeEventFactory events;

  std::vector<SharedScratchDevice*> before;
  before.reserve(kDevices);
  for (int id = 0; id < kDevices; ++id) {
    SharedScratchDevice& dev = pool.get(id);
    // Give each slot something, so the reset has work to do on all of them rather
    // than skipping past empty ones.
    ASSERT_NE(call(dev, alloc, 1024), nullptr);
    ASSERT_NE(shared_scratch_claim_event(dev, std::ref(events)).event, nullptr);
    before.push_back(&dev);
  }

  int disposed = 0;
  reset_shared_scratch_pool_slots(pool, [&disposed](int, void*, cudaEvent_t, cudaStream_t) { ++disposed; });
  ASSERT_EQ(disposed, kDevices);

  int moved = 0;
  for (int id = 0; id < kDevices; ++id) {
    if (&pool.get(id) != before[static_cast<std::size_t>(id)]) {
      ++moved;
    }
  }
  EXPECT_EQ(moved, 0) << moved << " of " << kDevices
                      << " entries moved, so a reference the registry handed out before the reset names freed memory "
                         "or another device's slot";
}

// The reset's disposer frees device memory, and a device-wide free waits on
// everything queued on that device -- a parked host function included. Under
// either lock that would hold up whatever the lock covers: the device's own next
// claimant under the device lock, and every device's claimants under the
// registry's. So each slot is emptied under its own device lock, and what came
// out of it is disposed of with that dropped too. The registry lock is dropped
// earlier still, after the snapshot of which slots exist; the case below it is
// what watches that.
TEST(SharedScratchPoolRegistry, ResetDisposesWithNoLockHeld) {
  constexpr int kUntouchedDevice = 99;
  SharedScratchPool pool;
  FakeAllocator alloc;

  SharedScratchDevice& dev0 = pool.get(0);
  ASSERT_NE(call(dev0, alloc, 1024), nullptr);

  std::atomic<bool> disposing{false};
  std::atomic<bool> reset_returned{false};
  std::atomic<bool> nothing_disposed{false};
  std::atomic<bool> claimed{false};
  std::atomic<bool> device_lock_free{false};
  std::thread claimer([&] {
    // The regression this case exists to catch is a reset that disposes of
    // nothing, and this wait is where that regression arrives. Waiting for
    // `disposing` alone would turn it into a target that hangs until Bazel kills
    // it, with no assertion to say what broke.
    const auto rendezvous_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!disposing.load() && !reset_returned.load() && std::chrono::steady_clock::now() < rendezvous_deadline) {
      std::this_thread::yield();
    }
    if (!disposing.load()) {
      nothing_disposed.store(true);
      return;
    }
    if (dev0.mu.try_lock()) {
      device_lock_free.store(true);
      dev0.mu.unlock();
    }
    pool.get(kUntouchedDevice);
    claimed.store(true);
  });

  // Read inside the disposer: the claim completes once the reset returns either
  // way, so only what was true while the disposer ran tells the two apart.
  bool claimed_during_dispose = false;
  reset_shared_scratch_pool_slots(pool, [&](int, void*, cudaEvent_t, cudaStream_t) {
    disposing.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!claimed.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    claimed_during_dispose = claimed.load();
  });
  reset_returned.store(true);
  claimer.join();

  ASSERT_FALSE(nothing_disposed.load()) << "the reset returned without disposing of the slot that holds a buffer, so "
                                           "there was no window in which to observe either lock";
  EXPECT_TRUE(device_lock_free.load()) << "the disposer ran holding the slot's own device lock, so a claim on that "
                                          "device waits for a free that waits on the whole device";
  EXPECT_TRUE(claimed_during_dispose) << "a lookup for a device the reset never touched could not complete while the "
                                         "disposer ran, so the registry's lock was held across it and one device's "
                                         "teardown blocks every other device";
}

// The other half of the same rule. The case above watches the disposer, which
// runs after the retry loop; this one watches the loop itself, which is where the
// reset spends its whole deadline whenever a slot's lock is held. Every pooled
// call on every device takes the registry lock to find its own slot, so a reset
// that held it across that loop would stop devices it never takes for as long as
// the loop runs.
//
// The lookup is timed in a loop rather than once, because the reset takes the
// registry lock at an instant this thread does not control and a single lookup
// could land before it and measure nothing. What the reset cost a device it never
// touched is the longest of them.
TEST(SharedScratchPoolRegistry, ALockedSlotDoesNotBlockLookupsForTheDevicesTheResetNeverTakes) {
  constexpr int kUntouchedDevice = 7;
  // Generous enough for a loaded machine and still far below what a reset holding
  // the registry lock across its retry spends, which is the whole deadline: what
  // is being told apart is milliseconds from seconds.
  const auto longest_acceptable_block = kResetLockWait / 5;

  SharedScratchPool pool;
  FakeAllocator alloc;
  SharedScratchDevice& leaked = pool.get(0);
  ASSERT_NE(call(leaked, alloc, 1024), nullptr);
  // Created before the reset, so what is timed below is a lookup and not an
  // insertion racing the reset's walk.
  pool.get(kUntouchedDevice);

  std::atomic<bool> holding{false};
  std::atomic<bool> release_it{false};
  std::thread holder([&] {
    leaked.mu.lock();
    holding.store(true);
    const auto deadline = std::chrono::steady_clock::now() + kResetLockWait + std::chrono::seconds{25};
    while (!release_it.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    leaked.mu.unlock();
  });
  while (!holding.load()) {
    std::this_thread::yield();
  }

  std::atomic<bool> reset_returned{false};
  std::size_t left_alone = 0;
  std::thread resetter([&] {
    left_alone = reset_shared_scratch_pool_slots(pool, [](int, void*, cudaEvent_t, cudaStream_t) {});
    reset_returned.store(true);
  });

  std::chrono::steady_clock::duration longest_block{};
  int lookups = 0;
  while (!reset_returned.load()) {
    const auto before = std::chrono::steady_clock::now();
    pool.get(kUntouchedDevice);
    const auto blocked = std::chrono::steady_clock::now() - before;
    ++lookups;
    if (blocked > longest_block) {
      longest_block = blocked;
    }
    // Enough of a gap that this loop does not starve the reset of the very lock
    // the case is about.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  resetter.join();
  release_it.store(true);
  holder.join();

  ASSERT_EQ(left_alone, 1u) << "the reset did not spend its deadline retrying the locked slot, so nothing below was "
                               "measured against a reset that was retrying";
  ASSERT_GT(lookups, 1) << "the reset was over before this thread had looked the untouched device up twice";
  EXPECT_LT(longest_block, longest_acceptable_block)
      << "a lookup for a device the reset never takes blocked "
      << std::chrono::duration_cast<std::chrono::milliseconds>(longest_block).count() << " ms of the reset's "
      << std::chrono::duration_cast<std::chrono::milliseconds>(kResetLockWait).count()
      << " ms deadline, so the registry lock is held across the retry loop and one slot the reset cannot take holds "
         "off every device";
}

// A device lock still held between cases is a claim that was never released --
// the defect the backend suite's own case hunts. Waiting for it would hang the
// fixture that resets before and after every case, so the run would end in a
// target timeout and the message naming the leak would never be printed. The
// reset reports the slot and leaves it instead.
//
// The holder below releases on its own deadline, so a reset that waits for the
// lock rather than reporting it ends this case in a failure rather than a hang.
TEST(SharedScratchPoolRegistry, ResetReportsALockedSlotRatherThanWaitingForIt) {
  constexpr std::chrono::seconds kHolderDeadline{30};

  SharedScratchPool pool;
  FakeAllocator alloc;

  // Two unlocked slots against one locked one, so the number the reset answers
  // with -- devices it could not take -- differs from the number it cleared. With
  // one of each the two are both 1 and a reset that reported the wrong one of them
  // would pass.
  SharedScratchDevice& leaked = pool.get(0);
  SharedScratchDevice& ordinary = pool.get(1);
  SharedScratchDevice& also_ordinary = pool.get(2);
  ASSERT_NE(call(leaked, alloc, 1024), nullptr);
  ASSERT_NE(call(ordinary, alloc, 2048), nullptr);
  ASSERT_NE(call(also_ordinary, alloc, 4096), nullptr);

  std::atomic<bool> holding{false};
  std::atomic<bool> release_it{false};
  std::thread holder([&] {
    leaked.mu.lock();
    holding.store(true);
    const auto deadline = std::chrono::steady_clock::now() + kHolderDeadline;
    while (!release_it.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    leaked.mu.unlock();
  });
  while (!holding.load()) {
    std::this_thread::yield();
  }

  int disposed = 0;
  const auto started = std::chrono::steady_clock::now();
  const std::size_t still_locked =
      reset_shared_scratch_pool_slots(pool, [&](int, void*, cudaEvent_t, cudaStream_t) { ++disposed; });
  const auto took = std::chrono::steady_clock::now() - started;
  release_it.store(true);
  holder.join();

  EXPECT_EQ(still_locked, 1u) << "the reset did not report the one device whose lock it could not take";
  EXPECT_LT(took, kHolderDeadline) << "the reset returned only once the lock was released, so it waited for a claim "
                                      "that a leak would never release";
  EXPECT_EQ(disposed, 2) << "the reset disposed of " << disposed
                         << " slots: it should hand back the two unlocked ones and leave the locked one alone";
  EXPECT_NE(leaked.buffer, nullptr) << "the reset emptied a slot whose lock it never held, so it handed the disposer a "
                                       "buffer a live claimant is still using";
  EXPECT_EQ(ordinary.buffer, nullptr) << "one locked slot stopped the reset clearing the others";
  EXPECT_EQ(also_ordinary.buffer, nullptr) << "one locked slot stopped the reset clearing the others";
}

// The budget covers the whole reset, so a leaked lock has to be prevented from
// spending it on behalf of the devices it shares the pool with. A device that is
// merely busy -- claims arriving and finishing, which is every device the pool is
// for -- must still be taken, and a single try_lock made once the leak has run the
// clock down only takes it if it happens to be idle at that one instant.
//
// So the neighbour below is locked when the reset starts, free for half a second
// in the middle, and locked again long before the deadline. A reset that retries
// takes it inside that window; one that spends its budget on the leak first and
// then tries once reports it as locked and never hands its buffer to the disposer
// -- which, in the fixture that resets between cases, means the next case starts
// on a device still holding what an earlier one grew.
//
// Which slot is which is decided by the map rather than by this case, and only a
// leak the reset reaches *before* the busy slot can cost it anything. So the order
// is read off a reset of the same three entries first -- entries are never erased,
// so the second walk takes them in the same order -- and the roles are handed out
// from that.
TEST(SharedScratchPoolRegistry, ALeakedLockDoesNotCostTheResetTheSlotsItSharesThePoolWith) {
  constexpr std::chrono::milliseconds kNeighbourFreeFrom{200};
  constexpr std::chrono::milliseconds kNeighbourBusyAgainFrom{700};
  const auto holder_deadline = kResetLockWait + std::chrono::seconds{25};

  SharedScratchPool pool;
  FakeAllocator alloc;
  // Three slots, so the number the reset answers with and the number it cleared
  // stay different figures: one leaked, one busy, one free throughout.
  for (const int device_id : {0, 1, 2}) {
    ASSERT_NE(call(pool.get(device_id), alloc, 1024), nullptr);
  }
  std::vector<int> walk_order;
  ASSERT_EQ(
      reset_shared_scratch_pool_slots(
          pool, [&](int id, void*, cudaEvent_t, cudaStream_t) { walk_order.push_back(id); }),
      0u)
      << "a reset of three unlocked slots reported one it could not take";
  ASSERT_EQ(walk_order.size(), 3u);

  SharedScratchDevice& leaked = pool.get(walk_order[0]);
  SharedScratchDevice& busy = pool.get(walk_order[1]);
  SharedScratchDevice& ordinary = pool.get(walk_order[2]);
  ASSERT_NE(call(leaked, alloc, 1024), nullptr);
  ASSERT_NE(call(busy, alloc, 2048), nullptr);
  ASSERT_NE(call(ordinary, alloc, 4096), nullptr);

  std::atomic<bool> both_held{false};
  std::atomic<bool> release_it{false};
  const auto started = std::chrono::steady_clock::now();
  std::thread holder([&] {
    leaked.mu.lock();
    busy.mu.lock();
    both_held.store(true);
    std::this_thread::sleep_until(started + kNeighbourFreeFrom);
    busy.mu.unlock();
    std::this_thread::sleep_until(started + kNeighbourBusyAgainFrom);
    // Locked again well inside the deadline, so only a reset that looked during
    // the window above ever had it.
    busy.mu.lock();
    const auto deadline = std::chrono::steady_clock::now() + holder_deadline;
    while (!release_it.load() && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    busy.mu.unlock();
    leaked.mu.unlock();
  });
  while (!both_held.load()) {
    std::this_thread::yield();
  }

  int disposed = 0;
  const std::size_t still_locked =
      reset_shared_scratch_pool_slots(pool, [&](int, void*, cudaEvent_t, cudaStream_t) { ++disposed; });
  release_it.store(true);
  holder.join();

  EXPECT_EQ(still_locked, 1u) << "the reset reported " << still_locked
                              << " locked slots: only the leaked one is unreachable, and the busy one was free for "
                              << (kNeighbourBusyAgainFrom - kNeighbourFreeFrom).count() << " ms of the "
                              << std::chrono::duration_cast<std::chrono::milliseconds>(kResetLockWait).count()
                              << " ms it had to look";
  EXPECT_EQ(disposed, 2) << "the reset disposed of " << disposed
                         << " slots: the leaked one it must leave alone, and the busy one it must come back to rather "
                            "than give up on because another device's lock spent the budget";
  EXPECT_EQ(busy.buffer, nullptr) << "one leaked lock left a busy device's buffer resident, so the pool a case is "
                                     "reset for still holds what an earlier one grew";
  EXPECT_EQ(ordinary.buffer, nullptr) << "the reset left a slot nothing ever locked";
  EXPECT_NE(leaked.buffer, nullptr) << "the reset emptied a slot whose lock it never held";
}

TEST(SharedScratchPoolRegistry, AGrowthOnOneDeviceDoesNotBlockAClaimOnAnother) {
  SharedScratchPool pool;
  // One allocator per thread: the two claims share the registry and nothing else.
  FakeAllocator zero;
  FakeAllocator one;

  std::promise<void> entered_alloc;
  std::promise<void> leave_alloc;
  std::future<void> entered = entered_alloc.get_future();
  std::shared_future<void> leave = leave_alloc.get_future().share();

  SharedScratchDevice& dev0 = pool.get(0);
  std::thread grower([&] {
    std::lock_guard<std::mutex> lk(dev0.mu);
    RetiredScratch retired;
    shared_scratch_get_or_grow(
        dev0,
        4096,
        [&](std::size_t bytes) {
          entered_alloc.set_value();
          leave.wait();
          return zero.alloc(bytes);
        },
        retired);
  });
  // The cap matters as much as the wait: a growth that takes the reuse path never
  // reaches its allocation, so nothing fires this promise and an uncapped wait
  // would hang the harness rather than fail the test.
  if (entered.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
    leave_alloc.set_value();
    grower.join();
    FAIL() << "the growth on device 0 never reached its allocation";
  }

  // Device 0's growth is stalled inside its allocation with device 0's lock held.
  // Without this the rest of the test would pass against any implementation.
  if (dev0.mu.try_lock()) {
    dev0.mu.unlock();
    ADD_FAILURE() << "device 0's lock was not held across its allocation";
  }

  auto claim = std::async(std::launch::async, [&] {
    SharedScratchDevice& dev1 = pool.get(1);
    std::lock_guard<std::mutex> lk(dev1.mu);
    RetiredScratch retired;
    return shared_scratch_get_or_grow(
        dev1, 2048, [&](std::size_t bytes) { return one.alloc(bytes); }, retired);
  });
  const bool served = claim.wait_for(std::chrono::seconds(10)) == std::future_status::ready;

  leave_alloc.set_value();
  grower.join();

  ASSERT_TRUE(served) << "a claim on device 1 waited for a growth on device 0";
  EXPECT_NE(claim.get(), nullptr);
  EXPECT_EQ(one.alloc_count(), 1);
}

TEST(SharedScratchPoolRegistry, ConcurrentLookupsKeepTheRegistryIntact) {
  // Every other test reaches the registry from one thread at a time, so the
  // registry's own lock is the one mechanism here that nothing else exercises:
  // without this test it can be deleted outright and the suite stays green.
  //
  // An unsynchronized std::unordered_map mutated from several threads has no
  // defined behaviour, so this cannot assert on a specific corruption. It
  // hammers the lookup and then asks the two questions the corruption answers
  // wrongly: is every id still where the race left it, and did any two ids land
  // on one entry. Each round is an independent chance to observe that; the
  // rounds are what make a miss unlikely rather than the assertions.
  constexpr int kThreads = 4;
  constexpr int kPerThread = 4000;
  constexpr int kRounds = 8;

  for (int round = 0; round < kRounds; ++round) {
    SharedScratchPool pool;
    std::vector<std::vector<SharedScratchDevice*>> seen(kThreads);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        std::vector<SharedScratchDevice*> mine;
        mine.reserve(kPerThread);
        // Rehashing is what corrupts an unsynchronized map, and it happens on a
        // handful of the inserts in a round, so the threads have to be inside
        // their loops at the same time.
        ready.fetch_add(1);
        while (!go.load()) {
        }
        for (int i = 0; i < kPerThread; ++i) {
          mine.push_back(&pool.get(t * kPerThread + i));
        }
        seen[t] = std::move(mine);
      });
    }
    while (ready.load() < kThreads) {
    }
    go.store(true);
    for (std::thread& t : threads) {
      t.join();
    }

    std::set<SharedScratchDevice*> distinct;
    for (int t = 0; t < kThreads; ++t) {
      ASSERT_EQ(seen[t].size(), static_cast<std::size_t>(kPerThread));
      for (int i = 0; i < kPerThread; ++i) {
        const int id = t * kPerThread + i;
        ASSERT_EQ(&pool.get(id), seen[t][i]) << "device " << id << " in round " << round;
        distinct.insert(seen[t][i]);
      }
    }
    ASSERT_EQ(distinct.size(), static_cast<std::size_t>(kThreads * kPerThread)) << "round " << round;
  }
}

#if defined(__unix__)

// scratch_pool() hands back a registry that is allocated once and never
// destroyed. The alternative -- an object with static storage duration -- is
// destroyed after main returns, which destroys the registry's mutex, every
// device's mutex and the map nodes a live reference points into, while a thread
// anywhere between the lookup and the release of its device lock still holds
// them.
//
// Nothing gtest asserts inside this process can see that: it happens after the
// last test has finished. So these run in forked children that put a thread
// inside the pooled path and then exit, and read the status the child died with.
//
// Probabilistic in one direction only. A child that is torn down under a live
// claimant does not have to crash, so the count below is a floor on how often it
// would; a child that is not torn down under one cannot crash at all, so a
// single non-zero status is a real failure and not a flake.
constexpr int kTeardownChildren = 24;

// What a child exits with when it could not get as far as the state under test --
// starting a thread, two dozen forks in, is the way that happens. Distinct from
// zero and from the teardown crash this case counts, so a run that could not set
// itself up is not read as evidence either way.
constexpr int kChildCouldNotStart = 121;

// Long enough that the children below -- a thread, a 30 ms sleep and exit() --
// are nowhere near it.
constexpr std::chrono::seconds kChildDeadline{30};

// Reaps `child`, killing it if it overruns `kChildDeadline`; returns false when
// it had to. Without a deadline a wedged child parks the parent here until the
// whole target times out, which reports as a timeout on the binary rather than as
// this case failing. The deadline only delivers that if the whole reaping loop
// fits inside the target's timeout -- kTeardownChildren children at kChildDeadline
// each is 720 s -- which is why the BUILD file gives this target the long (900 s)
// timeout rather than leaving it Bazel's 300 s default. An interrupted wait is
// retried, since a signal can arrive at any point and says nothing about the
// child.
bool reap_child(pid_t child, int& status) {
  const auto deadline = std::chrono::steady_clock::now() + kChildDeadline;
  for (;;) {
    const pid_t reaped = waitpid(child, &status, WNOHANG);
    if (reaped == child) {
      return true;
    }
    if (reaped == -1 && errno != EINTR) {
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      kill(child, SIGKILL);
      // Reaped even so. A zombie is not inherited by anything -- it stays this
      // process's child until this process reaps it or exits -- so what leaving it
      // costs is a process-table entry per overrun, held for the rest of the run.
      while (waitpid(child, &status, 0) == -1 && errno == EINTR) {
      }
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

[[noreturn]] void hold_the_pool_open_then_exit() {
  static std::atomic<bool> inside{false};
  std::thread claimant([] {
    for (unsigned i = 0;; ++i) {
      SharedScratchDevice& dev = scratch_pool().get(static_cast<int>(i % 8));
      std::lock_guard<std::mutex> lk(dev.mu);
      dev.capacity += 1;
      inside.store(true);
    }
  });
  // Never joined: the point is for it to still be in there at teardown.
  claimant.detach();
  while (!inside.load()) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  // exit() and not _exit(): static destruction is the event under test.
  std::exit(0);
}

TEST(SharedScratchPoolRegistry, TheProcessPoolOutlivesStaticDestructionUnderALiveClaimant) {
  int killed_by_signal = 0;
  int exited_nonzero = 0;
  int could_not_start = 0;
  int overran = 0;
  for (int i = 0; i < kTeardownChildren; ++i) {
    // Anything gtest has buffered would otherwise be written twice, once by each
    // side of the fork.
    std::fflush(nullptr);
    const pid_t child = fork();
    ASSERT_NE(child, -1) << "fork failed: " << std::strerror(errno);
    if (child == 0) {
      // Nothing may leave the child by any route but exit. Left to unwind, a
      // failed thread construction returns into the test body and the child
      // carries on through the rest of the binary as a second gtest process,
      // forking children of its own and writing over the parent's output.
      try {
        hold_the_pool_open_then_exit();
      } catch (...) {
        _exit(kChildCouldNotStart);
      }
    }
    int status = 0;
    if (!reap_child(child, status)) {
      ++overran;
      continue;
    }
    if (WIFSIGNALED(status)) {
      ++killed_by_signal;
    } else if (WEXITSTATUS(status) == kChildCouldNotStart) {
      ++could_not_start;
    } else if (WEXITSTATUS(status) != 0) {
      ++exited_nonzero;
    }
  }

  EXPECT_EQ(overran, 0) << overran << " of " << kTeardownChildren << " children were still running after "
                        << kChildDeadline.count()
                        << " seconds and were killed, so they neither reached teardown nor reported anything about it";
  EXPECT_EQ(could_not_start, 0) << could_not_start << " of " << kTeardownChildren
                                << " children could not start their claimant thread, so those runs never put anything "
                                   "inside the pool and say nothing about what teardown does to a live claimant";
  EXPECT_EQ(killed_by_signal, 0) << killed_by_signal << " of " << kTeardownChildren
                                 << " children died on a signal at exit with a thread still inside the pool, so the "
                                    "registry is being destroyed out from under a live claimant";
  EXPECT_EQ(exited_nonzero, 0) << exited_nonzero << " of " << kTeardownChildren
                               << " children exited non-zero at teardown with a thread still inside the pool";
}

#endif // defined(__unix__)

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
