#include <atomic>
#include <chrono>
#include <filesystem>
#include <system_error>
#include <thread>

#include "core/util/file_lock.h"
#include "gtest/gtest.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace {

using torch_tensorrt::core::util::FileLock;
using namespace std::chrono_literals;

// Minimal unique-path helper that also deletes the lock file on scope exit so tests stay
// hermetic. Tests rely on inputs being fresh; we don't share state across cases.
struct LockPath {
  std::filesystem::path path;

  explicit LockPath(const std::string& tag) {
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    path =
        std::filesystem::temp_directory_path() / ("torch_trt_file_lock_" + tag + "_" + std::to_string(stamp) + ".lock");
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  ~LockPath() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

} // namespace

TEST(FileLock, ConstructCreatesLockFile) {
  LockPath p("ctor");
  ASSERT_FALSE(std::filesystem::exists(p.path));
  FileLock lock(p.path);
  EXPECT_TRUE(std::filesystem::exists(p.path));
  EXPECT_FALSE(lock.owns_lock());
}

TEST(FileLock, ExclusiveBlocksExclusive) {
  LockPath p("excl_excl");
  FileLock a(p.path);
  FileLock b(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));
  EXPECT_FALSE(b.try_lock(FileLock::Mode::Exclusive));
  EXPECT_TRUE(a.owns_lock());
  EXPECT_FALSE(b.owns_lock());
}

TEST(FileLock, SharedAllowsShared) {
  LockPath p("shared_shared");
  FileLock a(p.path);
  FileLock b(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Shared));
  EXPECT_TRUE(b.try_lock(FileLock::Mode::Shared));
  EXPECT_TRUE(a.owns_lock());
  EXPECT_TRUE(b.owns_lock());
}

TEST(FileLock, SharedBlocksExclusive) {
  LockPath p("shared_excl");
  FileLock a(p.path);
  FileLock b(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Shared));
  EXPECT_FALSE(b.try_lock(FileLock::Mode::Exclusive));
}

TEST(FileLock, ExclusiveBlocksShared) {
  LockPath p("excl_shared");
  FileLock a(p.path);
  FileLock b(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));
  EXPECT_FALSE(b.try_lock(FileLock::Mode::Shared));
}

TEST(FileLock, TryLockForTimesOut) {
  LockPath p("timeout");
  FileLock a(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));

  FileLock b(p.path);
  constexpr auto kTimeout = 200ms;
  auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(b.try_lock_for(FileLock::Mode::Exclusive, kTimeout));
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_GE(elapsed, kTimeout);
  // Generous upper bound -- poll cadence is 50ms and CI is sometimes slow.
  EXPECT_LT(elapsed, kTimeout + 1s);
}

TEST(FileLock, TryLockForSucceedsAfterRelease) {
  LockPath p("succeed");
  FileLock a(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));

  std::atomic<bool> released{false};
  std::thread releaser([&] {
    std::this_thread::sleep_for(150ms);
    a.unlock();
    released.store(true, std::memory_order_release);
  });

  FileLock b(p.path);
  bool acquired = b.try_lock_for(FileLock::Mode::Exclusive, 5s);
  releaser.join();
  EXPECT_TRUE(acquired);
  EXPECT_TRUE(released.load(std::memory_order_acquire));
  EXPECT_TRUE(b.owns_lock());
}

TEST(FileLock, MoveTransfersOwnership) {
  LockPath p("move");
  FileLock a(p.path);
  ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));
  EXPECT_TRUE(a.owns_lock());

  FileLock moved = std::move(a);
  EXPECT_TRUE(moved.owns_lock());
  EXPECT_FALSE(a.owns_lock());

  // Even though `moved` holds the lock, a new instance should still see it as held.
  FileLock c(p.path);
  EXPECT_FALSE(c.try_lock(FileLock::Mode::Exclusive));
}

TEST(FileLock, DestructorReleases) {
  LockPath p("dtor");
  {
    FileLock a(p.path);
    ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));
  }
  // After scope exit, another locker should acquire freely.
  FileLock b(p.path);
  EXPECT_TRUE(b.try_lock(FileLock::Mode::Exclusive));
}

TEST(FileLock, DoesNotUnlinkOnRelease) {
  LockPath p("no_unlink");
  {
    FileLock a(p.path);
    ASSERT_TRUE(a.try_lock(FileLock::Mode::Exclusive));
  }
  EXPECT_TRUE(std::filesystem::exists(p.path));
}

TEST(FileLock, OpenFailureThrows) {
  // Construct a path whose parent directory does not exist; open() / CreateFileW must
  // fail and propagate as std::system_error.
  auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path bad =
      std::filesystem::temp_directory_path() / ("torch_trt_no_dir_" + std::to_string(stamp)) / "x.lock";
  EXPECT_THROW(FileLock{bad}, std::system_error);
}

#ifndef _WIN32
TEST(FileLock, InteropFlockNamespaceMatch) {
  // Take an exclusive flock on the file directly with a separately-opened fd, then
  // assert FileLock::try_lock(Exclusive) sees the contention. Proves we are in the same
  // BSD-flock namespace filelock uses (and not the independent fcntl namespace).
  LockPath p("interop_flock");
  // Touch the file so flock has something to open (FileLock would create it via O_CREAT,
  // but the raw flock here uses O_RDWR only -- create explicitly).
  int seed_fd = ::open(p.path.c_str(), O_RDWR | O_CREAT, 0644);
  ASSERT_GE(seed_fd, 0);
  ASSERT_EQ(::close(seed_fd), 0);

  int raw_fd = ::open(p.path.c_str(), O_RDWR);
  ASSERT_GE(raw_fd, 0);
  ASSERT_EQ(::flock(raw_fd, LOCK_EX | LOCK_NB), 0);

  FileLock lock(p.path);
  EXPECT_FALSE(lock.try_lock(FileLock::Mode::Exclusive));
  EXPECT_FALSE(lock.try_lock(FileLock::Mode::Shared));

  ASSERT_EQ(::flock(raw_fd, LOCK_UN), 0);
  ::close(raw_fd);
}
#endif
