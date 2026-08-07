/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Pins the caller-stream properties the TensorRT backend relies on. The backend
// derives both the stream it enqueues on and whether it may return with work
// still in flight from one getCallerStream() read, so a change to any property
// below would silently change delegate behavior.

#include <cuda_runtime.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <gtest/gtest.h>

#include <optional>
#include <thread>

namespace {

namespace cuda = executorch::extension::cuda;

cudaStream_t sentinel(int index) {
  alignas(alignof(std::max_align_t)) static char storage[2][alignof(std::max_align_t)];
  return reinterpret_cast<cudaStream_t>(&storage[index]);
}

TEST(CallerStreamTest, NoGuardLeavesSelectionEmpty) {
  EXPECT_FALSE(cuda::getCallerStream().has_value());
}

TEST(CallerStreamTest, GuardSelectsStreamAndRestoresOnExit) {
  const cudaStream_t stream = sentinel(0);
  {
    cuda::CallerStreamGuard guard(stream);
    EXPECT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(stream));
  }
  EXPECT_FALSE(cuda::getCallerStream().has_value());
}

// The backend treats "a stream was selected" and "which stream" as separate
// answers, so an explicit null must stay engaged rather than read as no
// selection. This preserves the pre-refactor (stream, is_set) encoding.
TEST(CallerStreamTest, ExplicitNullStreamStaysEngaged) {
  {
    cuda::CallerStreamGuard guard(nullptr);
    const auto selected = cuda::getCallerStream();
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(*selected, nullptr);
  }
  EXPECT_FALSE(cuda::getCallerStream().has_value());
}

// cudaStreamPerThread names the same stream the backend falls back to, yet
// selecting it explicitly must still read as a caller selection.
TEST(CallerStreamTest, ExplicitPerThreadStreamStaysEngaged) {
  cuda::CallerStreamGuard guard(cudaStreamPerThread);
  const auto selected = cuda::getCallerStream();
  ASSERT_TRUE(selected.has_value());
  EXPECT_EQ(*selected, cudaStreamPerThread);
}

TEST(CallerStreamTest, NestedGuardsRestoreOuterSelection) {
  const cudaStream_t outer = sentinel(0);
  const cudaStream_t inner = sentinel(1);

  {
    cuda::CallerStreamGuard outer_guard(outer);
    EXPECT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(outer));
    {
      cuda::CallerStreamGuard inner_guard(inner);
      EXPECT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(inner));
      {
        cuda::CallerStreamGuard null_guard(nullptr);
        EXPECT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(nullptr));
      }
      EXPECT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(inner));
    }
    EXPECT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(outer));
  }
  EXPECT_FALSE(cuda::getCallerStream().has_value());
}

// The backend runs one handle per thread, so one thread's guard must not be
// visible to another.
TEST(CallerStreamTest, SelectionIsPerThread) {
  const cudaStream_t stream = sentinel(0);
  cuda::CallerStreamGuard guard(stream);
  ASSERT_EQ(cuda::getCallerStream(), std::optional<cudaStream_t>(stream));

  std::optional<cudaStream_t> observed_in_worker;
  bool worker_saw_selection = true;
  std::thread worker([&] {
    const auto selected = cuda::getCallerStream();
    worker_saw_selection = selected.has_value();
    observed_in_worker = selected;
  });
  worker.join();

  EXPECT_FALSE(worker_saw_selection);
  EXPECT_FALSE(observed_in_worker.has_value());
}

} // namespace
