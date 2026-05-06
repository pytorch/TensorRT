#include "core/util/file_lock.h"

#include <chrono>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace torch_tensorrt {
namespace core {
namespace util {

namespace {

constexpr std::chrono::milliseconds kPollInterval{50};

// Platform-isolated operations on detail::LockHandle. The #ifdefs live exclusively
// inside these helpers so call sites in the FileLock class stay platform-independent.

void open_handle(detail::LockHandle& h, const std::filesystem::path& path) {
#ifdef _WIN32
  HANDLE w = ::CreateFileW(
      path.wstring().c_str(),
      GENERIC_READ | GENERIC_WRITE,
      FILE_SHARE_READ | FILE_SHARE_WRITE,
      nullptr,
      OPEN_ALWAYS,
      FILE_ATTRIBUTE_NORMAL,
      nullptr);
  if (w == INVALID_HANDLE_VALUE) {
    throw std::system_error(
        static_cast<int>(::GetLastError()),
        std::system_category(),
        "FileLock: CreateFileW failed for " + path.string());
  }
  h.native = w;
#else
  int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) {
    throw std::system_error(errno, std::generic_category(), "FileLock: open failed for " + path.string());
  }
  h.native = fd;
#endif
}

void close_handle(detail::LockHandle& h) noexcept {
#ifdef _WIN32
  if (h.native != nullptr) {
    ::CloseHandle(h.native);
    h.native = nullptr;
  }
#else
  if (h.native >= 0) {
    ::close(h.native);
    h.native = -1;
  }
#endif
}

bool try_acquire(detail::LockHandle& h, FileLock::Mode mode, bool blocking) {
#ifdef _WIN32
  DWORD flags = (mode == FileLock::Mode::Exclusive) ? LOCKFILE_EXCLUSIVE_LOCK : 0;
  if (!blocking) {
    flags |= LOCKFILE_FAIL_IMMEDIATELY;
  }
  OVERLAPPED ovl{};
  if (::LockFileEx(h.native, flags, 0, 1, 0, &ovl)) {
    return true;
  }
  DWORD err = ::GetLastError();
  if (!blocking && err == ERROR_LOCK_VIOLATION) {
    return false;
  }
  throw std::system_error(static_cast<int>(err), std::system_category(), "FileLock: LockFileEx failed");
#else
  int operation = (mode == FileLock::Mode::Exclusive) ? LOCK_EX : LOCK_SH;
  if (!blocking) {
    operation |= LOCK_NB;
  }
  while (true) {
    if (::flock(h.native, operation) == 0) {
      return true;
    }
    int err = errno;
    if (err == EINTR) {
      continue;
    }
    if (!blocking && (err == EWOULDBLOCK || err == EAGAIN)) {
      return false;
    }
    throw std::system_error(err, std::generic_category(), "FileLock: flock failed");
  }
#endif
}

void release_handle(detail::LockHandle& h) noexcept {
#ifdef _WIN32
  OVERLAPPED ovl{};
  ::UnlockFileEx(h.native, 0, 1, 0, &ovl);
#else
  while (::flock(h.native, LOCK_UN) == -1 && errno == EINTR) {
  }
#endif
}

} // namespace

FileLock::FileLock(std::filesystem::path lock_path) : path_(std::move(lock_path)) {
  open_handle(handle_, path_);
}

FileLock::~FileLock() noexcept {
  if (owned_) {
    release_handle(handle_);
  }
  close_handle(handle_);
}

FileLock::FileLock(FileLock&& other) noexcept : FileLock() {
  swap(*this, other);
}

FileLock& FileLock::operator=(FileLock&& other) noexcept {
  FileLock tmp(std::move(other));
  swap(*this, tmp);
  return *this;
}

void swap(FileLock& a, FileLock& b) noexcept {
  using std::swap;
  swap(a.handle_, b.handle_);
  swap(a.owned_, b.owned_);
  swap(a.path_, b.path_);
}

void FileLock::lock(Mode mode) {
  try_acquire(handle_, mode, /*blocking=*/true);
  owned_ = true;
}

bool FileLock::try_lock(Mode mode) {
  if (try_acquire(handle_, mode, /*blocking=*/false)) {
    owned_ = true;
    return true;
  }
  return false;
}

bool FileLock::try_lock_for(Mode mode, std::chrono::milliseconds timeout) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    if (try_lock(mode)) {
      return true;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(kPollInterval);
  }
}

void FileLock::unlock() noexcept {
  if (!owned_) {
    return;
  }
  release_handle(handle_);
  owned_ = false;
}

bool FileLock::owns_lock() const noexcept {
  return owned_;
}

} // namespace util
} // namespace core
} // namespace torch_tensorrt
