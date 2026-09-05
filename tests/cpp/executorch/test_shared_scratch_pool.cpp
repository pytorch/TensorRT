// Pins the shared scratch pool helper: its grow, reuse and per-device policy and
// its enqueue-handoff rule, driven over fakes so no CUDA device is needed.
//
// This exercises the helper, not the backend: it does not link the delegate, so
// it cannot catch the delegate calling the helper wrongly or ceasing to call it.
// test_shared_scratch_backend covers that, and needs a GPU to do it.

#include "torch_tensorrt/executorch/SharedScratchPool.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
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

// Stands in for the backend: passes the allocator through and records whatever
// the call retired, the way execute() hands a retired buffer to its claim.
void* call(SharedScratchDevice& dev, FakeAllocator& a, std::size_t need, std::size_t& out_size) {
  RetiredScratch retired;
  void* const p = shared_scratch_get_or_grow(
      dev, need, out_size, [&a](std::size_t bytes) { return a.alloc(bytes); }, retired);
  if (retired.buffer != nullptr) {
    a.retire(retired.buffer, retired.wait_for);
  }
  return p;
}

TEST(SharedScratchPool, FirstRequestAllocatesExactSize) {
  SharedScratchDevice dev;
  FakeAllocator a;
  std::size_t out = 0;

  void* p = call(dev, a, /*need=*/1024, out);

  EXPECT_NE(p, nullptr);
  EXPECT_EQ(out, 1024u);
  ASSERT_EQ(a.alloc_count(), 1);
  EXPECT_EQ(a.alloc_sizes[0], 1024u);
  EXPECT_TRUE(a.retirements.empty());
}

TEST(SharedScratchPool, ReusesWhenExistingBufferIsLargeEnough) {
  SharedScratchDevice dev;
  FakeAllocator a;
  std::size_t out = 0;

  void* first = call(dev, a, 4096, out);
  // A smaller and an equal request must both reuse the same buffer (no realloc).
  // The smaller one reports into a fresh out2, so what the reuse path writes is
  // asserted rather than what the first call left in `out`.
  std::size_t out2 = 0;
  void* second = call(dev, a, 1000, out2);
  void* third = call(dev, a, 4096, out);

  EXPECT_EQ(second, first);
  EXPECT_EQ(third, first);
  EXPECT_EQ(out, 4096u);
  // Reuse reports the buffer's capacity, not the smaller amount asked for.
  EXPECT_EQ(out2, 4096u);
  EXPECT_EQ(a.alloc_count(), 1);
  EXPECT_TRUE(a.retirements.empty());
}

TEST(SharedScratchPool, GrowsMonotonicallyToMaxAndRetiresOldBuffer) {
  SharedScratchDevice dev;
  FakeAllocator a;
  std::size_t out = 0;

  void* small = call(dev, a, 1024, out);
  void* big = call(dev, a, 8192, out);

  EXPECT_NE(big, small);
  EXPECT_EQ(out, 8192u);
  ASSERT_EQ(a.alloc_count(), 2);
  EXPECT_EQ(a.alloc_sizes[1], 8192u);
  ASSERT_EQ(a.retirements.size(), 1u);
  EXPECT_EQ(a.retirements[0].first, small);

  // A subsequent smaller request reuses the grown buffer -- pool never shrinks.
  void* reuse = call(dev, a, 512, out);
  EXPECT_EQ(reuse, big);
  EXPECT_EQ(out, 8192u);
  EXPECT_EQ(a.alloc_count(), 2);
}

TEST(SharedScratchPool, GrowRetiresTheOldBufferWithTheEventToWaitOn) {
  SharedScratchDevice dev;
  FakeAllocator a;
  FakeEventFactory events;
  std::size_t out = 0;

  void* small = call(dev, a, 1024, out);
  ASSERT_NE(small, nullptr);

  // An enqueue against `small` has been submitted and recorded, so its
  // retirement has something specific to outlive.
  const SharedScratchHandoff handoff = shared_scratch_claim_event(dev, std::ref(events));
  ASSERT_EQ(shared_scratch_mark_in_flight(dev), handoff.event);

  ASSERT_NE(call(dev, a, 8192, out), nullptr);

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
  std::size_t out = 0;

  void* small = call(dev, a, 1024, out);
  ASSERT_NE(small, nullptr);
  // The slot has an event, but nothing has been recorded on it: claiming the
  // handoff is not the same as enqueueing against the buffer.
  ASSERT_NE(shared_scratch_claim_event(dev, std::ref(events)).event, nullptr);

  ASSERT_NE(call(dev, a, 8192, out), nullptr);

  ASSERT_EQ(a.retirements.size(), 1u);
  EXPECT_EQ(a.retirements[0].first, small);
  EXPECT_EQ(a.retirements[0].second, nullptr);
}

TEST(SharedScratchPool, AllocationFailureLeavesExistingBufferUntouched) {
  SharedScratchDevice dev;
  FakeAllocator a;
  std::size_t out = 0;

  void* first = call(dev, a, 1024, out);
  ASSERT_NE(first, nullptr);

  // A growth whose allocation fails must return nullptr and keep the old buffer,
  // so the caller can surface the error without corrupting the pool.
  a.fail_next = true;
  std::size_t out2 = 0;
  void* failed = call(dev, a, 8192, out2);
  EXPECT_EQ(failed, nullptr);
  EXPECT_TRUE(a.retirements.empty());

  // The device still holds the original buffer and serves it on the next request.
  void* again = call(dev, a, 1024, out);
  EXPECT_EQ(again, first);
  EXPECT_EQ(out, 1024u);
}

TEST(SharedScratchPool, FirstAllocationFailureReturnsNullAndStoresNothing) {
  SharedScratchDevice dev;
  FakeAllocator a;
  std::size_t out = 0;

  a.fail_next = true;
  void* p = call(dev, a, 1024, out);
  EXPECT_EQ(p, nullptr);
  EXPECT_EQ(dev.buffer, nullptr);
  EXPECT_EQ(dev.capacity, 0u);

  // Nothing stored: a later successful request allocates fresh.
  void* q = call(dev, a, 1024, out);
  EXPECT_NE(q, nullptr);
  EXPECT_EQ(a.alloc_count(), 1);
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
// The registry that owns one entry per device.
// ---------------------------------------------------------------------------

TEST(SharedScratchPoolRegistry, KeepsAnIndependentBufferPerDevice) {
  SharedScratchPool pool;
  FakeAllocator a;
  std::size_t out = 0;

  void* dev0 = call(pool.get(0), a, 2048, out);
  void* dev1 = call(pool.get(1), a, 2048, out);

  EXPECT_NE(dev0, dev1);
  EXPECT_EQ(a.alloc_count(), 2);
  EXPECT_TRUE(a.retirements.empty());

  // Growing device 1 must not touch device 0's buffer.
  void* dev1_big = call(pool.get(1), a, 9000, out);
  void* dev0_again = call(pool.get(0), a, 2048, out);
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
    std::size_t out = 0;
    RetiredScratch retired;
    shared_scratch_get_or_grow(
        dev0,
        4096,
        out,
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
    std::size_t out = 0;
    RetiredScratch retired;
    return shared_scratch_get_or_grow(
        dev1, 2048, out, [&](std::size_t bytes) { return one.alloc(bytes); }, retired);
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

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
