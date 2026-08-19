#include "torch_tensorrt/executorch/SharedScratchPool.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

// Fake device allocator: hands out distinct non-null pointers and records every
// allocation size and every released pointer, so tests can assert the pool's
// grow/reuse/per-device policy without a CUDA device.
struct FakeAllocator {
  std::vector<std::size_t> alloc_sizes;
  std::vector<void*> released;
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

  void release(void* p) {
    released.push_back(p);
  }

  int alloc_count() const {
    return static_cast<int>(alloc_sizes.size());
  }
};

using Pool = std::unordered_map<int, std::pair<void*, std::size_t>>;

void* call(Pool& pool, FakeAllocator& a, int device_id, std::size_t need, std::size_t& out_size) {
  return shared_scratch_get_or_grow(
      pool,
      device_id,
      need,
      out_size,
      [&a](std::size_t bytes) { return a.alloc(bytes); },
      [&a](void* p) { a.release(p); });
}

TEST(SharedScratchPool, FirstRequestAllocatesExactSize) {
  Pool pool;
  FakeAllocator a;
  std::size_t out = 0;

  void* p = call(pool, a, /*device_id=*/0, /*need=*/1024, out);

  EXPECT_NE(p, nullptr);
  EXPECT_EQ(out, 1024u);
  ASSERT_EQ(a.alloc_count(), 1);
  EXPECT_EQ(a.alloc_sizes[0], 1024u);
  EXPECT_TRUE(a.released.empty());
}

TEST(SharedScratchPool, ReusesWhenExistingBufferIsLargeEnough) {
  Pool pool;
  FakeAllocator a;
  std::size_t out = 0;

  void* first = call(pool, a, 0, 4096, out);
  // A smaller and an equal request must both reuse the same buffer (no realloc).
  // The smaller one reports into a fresh out2, so what the reuse path writes is
  // asserted rather than what the first call left in `out`.
  std::size_t out2 = 0;
  void* second = call(pool, a, 0, 1000, out2);
  void* third = call(pool, a, 0, 4096, out);

  EXPECT_EQ(second, first);
  EXPECT_EQ(third, first);
  EXPECT_EQ(out, 4096u);
  // Reuse reports the buffer's capacity, not the smaller amount asked for.
  EXPECT_EQ(out2, 4096u);
  EXPECT_EQ(a.alloc_count(), 1);
  EXPECT_TRUE(a.released.empty());
}

TEST(SharedScratchPool, GrowsMonotonicallyToMaxAndReleasesOldBuffer) {
  Pool pool;
  FakeAllocator a;
  std::size_t out = 0;

  void* small = call(pool, a, 0, 1024, out);
  void* big = call(pool, a, 0, 8192, out);

  EXPECT_NE(big, small);
  EXPECT_EQ(out, 8192u);
  ASSERT_EQ(a.alloc_count(), 2);
  EXPECT_EQ(a.alloc_sizes[1], 8192u);
  ASSERT_EQ(a.released.size(), 1u);
  EXPECT_EQ(a.released[0], small);

  // A subsequent smaller request reuses the grown buffer -- pool never shrinks.
  void* reuse = call(pool, a, 0, 512, out);
  EXPECT_EQ(reuse, big);
  EXPECT_EQ(out, 8192u);
  EXPECT_EQ(a.alloc_count(), 2);
}

TEST(SharedScratchPool, KeepsIndependentBufferPerDevice) {
  Pool pool;
  FakeAllocator a;
  std::size_t out = 0;

  void* dev0 = call(pool, a, /*device_id=*/0, 2048, out);
  void* dev1 = call(pool, a, /*device_id=*/1, 2048, out);

  EXPECT_NE(dev0, dev1);
  EXPECT_EQ(a.alloc_count(), 2);
  EXPECT_TRUE(a.released.empty());

  // Growing device 1 must not touch device 0's buffer.
  void* dev1_big = call(pool, a, 1, 9000, out);
  void* dev0_again = call(pool, a, 0, 2048, out);
  EXPECT_NE(dev1_big, dev1);
  EXPECT_EQ(dev0_again, dev0);
  ASSERT_EQ(a.released.size(), 1u);
  EXPECT_EQ(a.released[0], dev1);
}

TEST(SharedScratchPool, AllocationFailureLeavesExistingSlotUntouched) {
  Pool pool;
  FakeAllocator a;
  std::size_t out = 0;

  void* first = call(pool, a, 0, 1024, out);
  ASSERT_NE(first, nullptr);

  // A growth whose allocation fails must return nullptr and keep the old buffer,
  // so the caller can surface the error without corrupting the pool.
  a.fail_next = true;
  std::size_t out2 = 0;
  void* failed = call(pool, a, 0, 8192, out2);
  EXPECT_EQ(failed, nullptr);
  EXPECT_TRUE(a.released.empty());

  // The pool still holds the original buffer and serves it on the next request.
  void* again = call(pool, a, 0, 1024, out);
  EXPECT_EQ(again, first);
  EXPECT_EQ(out, 1024u);
}

TEST(SharedScratchPool, FirstAllocationFailureReturnsNullAndStoresNothing) {
  Pool pool;
  FakeAllocator a;
  std::size_t out = 0;

  a.fail_next = true;
  void* p = call(pool, a, 0, 1024, out);
  EXPECT_EQ(p, nullptr);

  // Nothing stored: a later successful request allocates fresh.
  void* q = call(pool, a, 0, 1024, out);
  EXPECT_NE(q, nullptr);
  EXPECT_EQ(a.alloc_count(), 1);
}

// ---------------------------------------------------------------------------
// Ordering the shared buffer's handoff from one enqueue to the next.
// ---------------------------------------------------------------------------

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

using Markers = std::unordered_map<int, SharedScratchMarker>;

TEST(SharedScratchHandoffTest, FirstUseCreatesTheSlotsEventAndWaitsForNothing) {
  Markers markers;
  FakeEventFactory events;

  const SharedScratchHandoff handoff = shared_scratch_claim_event(markers, /*device_id=*/0, std::ref(events));

  EXPECT_NE(handoff.event, nullptr);
  EXPECT_FALSE(handoff.needs_wait);
  EXPECT_EQ(events.created, 1);
}

TEST(SharedScratchHandoffTest, EveryUseAfterAnEnqueueWaitsOnTheSameEvent) {
  Markers markers;
  FakeEventFactory events;
  const SharedScratchHandoff first = shared_scratch_claim_event(markers, 0, std::ref(events));
  ASSERT_FALSE(first.needs_wait);

  EXPECT_EQ(shared_scratch_mark_in_flight(markers, 0), first.event);

  // Every later enqueue waits, however many there have been and whichever stream
  // each of them ran on: the marker records that the buffer was handed out, not
  // who it was handed to. Comparing stream handles instead would let a caller
  // through whenever its handle matched the recorded one, including when CUDA has
  // recycled that value for a different stream.
  const SharedScratchHandoff second = shared_scratch_claim_event(markers, 0, std::ref(events));
  EXPECT_TRUE(second.needs_wait);
  EXPECT_EQ(second.event, first.event);

  const SharedScratchHandoff third = shared_scratch_claim_event(markers, 0, std::ref(events));
  EXPECT_TRUE(third.needs_wait);
  EXPECT_EQ(third.event, first.event);

  // One event serves the slot for its whole life, so the wait never targets an
  // event some earlier enqueue was recorded on.
  EXPECT_EQ(events.created, 1);
}

TEST(SharedScratchHandoffTest, KeepsAnIndependentMarkerPerDevice) {
  Markers markers;
  FakeEventFactory events;
  const SharedScratchHandoff dev0 = shared_scratch_claim_event(markers, /*device_id=*/0, std::ref(events));
  ASSERT_EQ(shared_scratch_mark_in_flight(markers, 0), dev0.event);

  // Device 1 has its own buffer, so device 0's enqueue is nothing for it to wait
  // on, and it gets its own event.
  const SharedScratchHandoff dev1 = shared_scratch_claim_event(markers, /*device_id=*/1, std::ref(events));
  EXPECT_FALSE(dev1.needs_wait);
  EXPECT_NE(dev1.event, dev0.event);
  EXPECT_EQ(events.created, 2);

  // Marking device 1 does not make device 0 stop waiting, or the other way round.
  ASSERT_EQ(shared_scratch_mark_in_flight(markers, 1), dev1.event);
  EXPECT_TRUE(shared_scratch_claim_event(markers, 0, std::ref(events)).needs_wait);
  EXPECT_TRUE(shared_scratch_claim_event(markers, 1, std::ref(events)).needs_wait);
}

TEST(SharedScratchHandoffTest, EventCreationFailureIsReportedAndRetried) {
  Markers markers;
  FakeEventFactory events;

  events.fail_next = true;
  const SharedScratchHandoff failed = shared_scratch_claim_event(markers, 0, std::ref(events));
  EXPECT_EQ(failed.event, nullptr);
  EXPECT_FALSE(failed.needs_wait);

  // The failure leaves nothing behind, so the next call tries again and succeeds
  // rather than serving an unusable slot for the rest of the process.
  const SharedScratchHandoff retried = shared_scratch_claim_event(markers, 0, std::ref(events));
  EXPECT_NE(retried.event, nullptr);
  EXPECT_FALSE(retried.needs_wait);
  EXPECT_EQ(events.created, 1);
}

TEST(SharedScratchHandoffTest, ASlotWithNoEventIsNotMarked) {
  Markers markers;
  FakeEventFactory events;

  // Nothing can be recorded without an event, so nothing is claimed to have been.
  EXPECT_EQ(shared_scratch_mark_in_flight(markers, 0), nullptr);

  // Otherwise, once an event is finally created for the slot, the next caller
  // would wait on it believing an enqueue had been recorded on it that never was.
  EXPECT_FALSE(shared_scratch_claim_event(markers, 0, std::ref(events)).needs_wait);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
