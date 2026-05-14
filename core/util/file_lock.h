#pragma once

#include <chrono>
#include <filesystem>

namespace torch_tensorrt {
namespace core {
namespace util {

namespace detail {

// Platform-specific handle for the underlying lock file. The platform #ifdef is confined
// to this struct definition; all operations on it (defined in file_lock.cpp) take a
// LockHandle& and keep the platform branch internal.
struct LockHandle {
#ifdef _WIN32
  void* native = nullptr; // HANDLE; void* keeps <windows.h> out of this header.
#else
  int native = -1;
#endif
};

} // namespace detail

// Cross-platform RAII file lock matching filelock's lock-file convention so the C++ and
// Python torch-TRT runtimes can safely share a runtime cache path.
//
// Backend: Unix uses BSD flock(2); Windows uses LockFileEx on byte (0,1). The byte range
// and primitive are deliberately matched to filelock so a Python FileLock and a C++
// FileLock on the same .lock file conflict correctly across the runtime boundary.
//
// Usage rule: each thread / call site must construct its own FileLock. flock locks live
// per open file description, so two threads sharing one FileLock instance share one OFD
// and would not actually serialize against each other.
class FileLock {
 public:
  enum class Mode { Shared, Exclusive };

  // Opens (creates if needed) the lock file. Throws std::system_error on open failure.
  explicit FileLock(std::filesystem::path lock_path);
  ~FileLock() noexcept;

  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  FileLock(FileLock&&) noexcept;
  FileLock& operator=(FileLock&&) noexcept;

  friend void swap(FileLock& a, FileLock& b) noexcept;

  // Blocks until acquired. Throws std::system_error on hard error.
  void lock(Mode mode);

  // Returns false on contention; throws std::system_error on hard error.
  [[nodiscard]] bool try_lock(Mode mode);

  // Polls until acquired or timeout (50ms cadence). False on timeout, throws on hard
  // error. Always makes at least one acquire attempt regardless of timeout value.
  [[nodiscard]] bool try_lock_for(Mode mode, std::chrono::milliseconds timeout);

  void unlock() noexcept;
  [[nodiscard]] bool owns_lock() const noexcept;

 private:
  // Empty lock (no underlying file). Private because its only purpose is to serve as
  // the delegation target for the move ctor (which then swaps in the source's state).
  FileLock() noexcept = default;

  detail::LockHandle handle_;
  bool owned_ = false;
  std::filesystem::path path_;
};

} // namespace util
} // namespace core
} // namespace torch_tensorrt
