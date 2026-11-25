// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Buffers must outlive this object and must not resize while registered.
// Register once outside timing, then reuse the same CPU storage across calls.
class RegisteredHostMemory {
 public:
  using Buffer = std::pair<void*, size_t>;
  explicit RegisteredHostMemory(const std::vector<Buffer>& buffers) {
    int device = 0, supported = 0;
    check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, device));
    if (!supported)
      throw std::runtime_error("CUDA host registration is unsupported");
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      throw std::runtime_error("Cannot determine host page size");
    const auto page = static_cast<uintptr_t>(page_size);
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    for (const auto& buffer : buffers) {
      if (!buffer.first || !buffer.second)
        throw std::runtime_error("Cannot register an empty buffer");
      const auto first = reinterpret_cast<uintptr_t>(buffer.first);
      ranges.emplace_back(first / page * page, (first + buffer.second + page - 1) / page * page);
    }
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (const auto& range : ranges) {
      if (!merged.empty() && range.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, range.second);
      else
        merged.push_back(range);
    }
    pages_.reserve(merged.size());
    try {
      for (const auto& range : merged) {
        void* address = reinterpret_cast<void*>(range.first);
        check(cudaHostRegister(address, range.second - range.first, cudaHostRegisterDefault));
        pages_.push_back(address);
      }
    } catch (...) {
      for (void* address : pages_)
        cudaHostUnregister(address);
      throw;
    }
  }
  RegisteredHostMemory(const RegisteredHostMemory&) = delete;
  RegisteredHostMemory& operator=(const RegisteredHostMemory&) = delete;
  ~RegisteredHostMemory() {
    // Also drain outstanding work on an exception path before unregistration.
    cudaDeviceSynchronize();
    for (void* address : pages_)
      cudaHostUnregister(address);
  }

 private:
  static void check(cudaError_t error) {
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("Register host IO: ") + cudaGetErrorString(error));
  }
  std::vector<void*> pages_;
};
