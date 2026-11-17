// SPDX-License-Identifier: BSD-3-Clause
// Shared measurement boundary for the two native PBR applications.
#pragma once
#include <cuda_runtime.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace pbr {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
inline void cuda_check(cudaError_t code, const char* operation) {
  require(code == cudaSuccess,
          std::string(operation) + ": " + cudaGetErrorString(code));
}
struct Stream {
  cudaStream_t value{};
  Stream() { cuda_check(cudaStreamCreate(&value), "create stream"); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct DeviceBuffer {
  void* value{};
  explicit DeviceBuffer(size_t bytes) {
    cuda_check(cudaMalloc(&value, bytes), "allocate device buffer");
  }
  ~DeviceBuffer() { cudaFree(value); }
};
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;
  explicit Tensor(const fs::path& stem) {
    std::ifstream dimensions(stem.string() + ".shape");
    require(bool(dimensions), "Missing shape: " + stem.string());
    int64_t dimension;
    size_t count = 1;
    while (dimensions >> dimension) {
      require(dimension > 0 && static_cast<uint64_t>(dimension) <=
                                   std::numeric_limits<size_t>::max() /
                                       sizeof(float) / count,
              "Invalid tensor dimension");
      shape.push_back(dimension);
      count *= dimension;
    }
    require(dimensions.eof() && !shape.empty(),
            "Invalid shape: " + stem.string());
    auto path = stem.string() + ".bin";
    require(fs::file_size(path) == count * sizeof(float),
            "Fixture size mismatch: " + path);
    data.resize(count);
    std::ifstream file(path, std::ios::binary);
    require(bool(file.read(reinterpret_cast<char*>(data.data()), bytes())),
            "Cannot read fixture");
    require(std::all_of(data.begin(), data.end(),
                        [](float x) { return std::isfinite(x); }),
            "Nonfinite fixture: " + path);
  }
  size_t bytes() const { return data.size() * sizeof(float); }
};
struct Fixtures {
  std::vector<Tensor> inputs, expected;
  std::vector<std::vector<float>> outputs;
  Fixtures(const fs::path& directory, size_t ni, size_t no) {
    for (size_t i = 0; i < ni; ++i)
      inputs.emplace_back(directory / ("input_" + std::to_string(i)));
    for (size_t i = 0; i < no; ++i) {
      expected.emplace_back(directory / ("output_" + std::to_string(i)));
      outputs.emplace_back(expected.back().data.size());
    }
  }
};
struct PinnedIO {
  std::vector<void*> pages;
  explicit PinnedIO(Fixtures& f) {
    int supported = 0;
    cuda_check(
        cudaDeviceGetAttribute(&supported, cudaDevAttrHostRegisterSupported, 0),
        "host registration query");
    require(supported, "Host registration is not supported on this target");
    long page = sysconf(_SC_PAGESIZE);
    require(page > 0, "Cannot get page size");
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    auto add = [&](std::vector<float>& v) {
      auto first = reinterpret_cast<uintptr_t>(v.data());
      ranges.emplace_back(
          first / page * page,
          (first + v.size() * sizeof(float) + page - 1) / page * page);
    };
    for (auto& x : f.inputs) add(x.data);
    for (auto& x : f.outputs) add(x);
    std::sort(ranges.begin(), ranges.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> merged;
    for (auto range : ranges) {
      if (!merged.empty() && range.first <= merged.back().second)
        merged.back().second = std::max(merged.back().second, range.second);
      else
        merged.push_back(range);
    }
    // Several small input tensors can share an allocator page.
    for (auto range : merged) {
      auto p = reinterpret_cast<void*>(range.first);
      auto code = cudaHostRegister(p, range.second - range.first,
                                   cudaHostRegisterDefault);
      if (code != cudaSuccess) {
        for (auto old : pages) cudaHostUnregister(old);
        cuda_check(code, "register caller buffer");
      }
      pages.push_back(p);
    }
  }
  ~PinnedIO() {
    cudaDeviceSynchronize();
    for (auto p : pages) cudaHostUnregister(p);
  }
};
inline long long kib(const char* path, const char* key) {
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    std::string name, unit;
    long long value;
    if (fields >> name >> value >> unit && name == key && unit == "kB")
      return value;
  }
  throw std::runtime_error(std::string("Missing memory counter: ") + key);
}
struct Report {
  fs::path directory;
  std::ofstream times, memory, checks;
  explicit Report(const fs::path& path) : directory(path) {
    require(fs::create_directory(path), "Output must be a new directory");
    times.open(path / "latency.tsv");
    memory.open(path / "memory.tsv");
    checks.open(path / "comparisons.tsv");
    require(times && memory && checks, "Cannot open reports");
    times << std::setprecision(12) << "phase\titeration\twall_ms\n";
    memory << "phase\trss_kib\tprocess_hwm_kib\tpss_kib\tcuda_free_bytes\tcuda_"
              "total_bytes\n";
    checks << std::setprecision(12)
           << "phase\toutput\telements\tmax_abs_error\tmismatches\n";
  }
  void snapshot(const char* phase) {
    if (std::string(phase) == "cuda_initialized") {
      cudaDeviceProp properties{};
      int driver = 0, runtime = 0;
      cuda_check(cudaGetDeviceProperties(&properties, 0), "GPU properties");
      cuda_check(cudaDriverGetVersion(&driver), "driver version");
      cuda_check(cudaRuntimeGetVersion(&runtime), "runtime version");
      std::ofstream info(directory / "device.txt");
      info << "GPU: " << properties.name
           << "\nSMs: " << properties.multiProcessorCount
           << "\nCompute capability: " << properties.major << '.'
           << properties.minor << "\nCUDA driver: " << driver
           << "\nCUDA runtime: " << runtime << '\n';
      info.close();
      require(bool(info), "Cannot save GPU properties");
    }
    size_t free, total;
    cuda_check(cudaMemGetInfo(&free, &total), "query CUDA memory");
    memory << phase << '\t' << kib("/proc/self/status", "VmRSS:") << '\t'
           << kib("/proc/self/status", "VmHWM:") << '\t'
           << kib("/proc/self/smaps_rollup", "Pss:") << '\t' << free << '\t'
           << total << '\n';
    memory.flush();
    require(bool(memory), "Memory report failed");
  }
  void compare(const Fixtures& f, const char* phase) {
    size_t total_bad = 0;
    for (size_t i = 0; i < f.outputs.size(); ++i) {
      size_t bad = 0;
      double maximum = 0;
      for (size_t j = 0; j < f.outputs[i].size(); ++j) {
        double got = f.outputs[i][j], want = f.expected[i].data[j];
        double error = std::abs(got - want);
        if (!std::isfinite(got) || error > 3e-4 + 3e-4 * std::abs(want)) ++bad;
        if (std::isfinite(error)) maximum = std::max(maximum, error);
      }
      checks << phase << '\t' << i << '\t' << f.outputs[i].size() << '\t'
             << maximum << '\t' << bad << '\n';
      total_bad += bad;
    }
    checks.flush();
    require(checks && total_bad == 0,
            "Output mismatch; performance result invalid");
  }
  template <class Run>
  void measure(Fixtures& f, Run run) {
    auto timed = [&]() {
      auto start = Clock::now();
      run();
      return std::chrono::duration<double, std::milli>(Clock::now() - start)
          .count();
    };
    times << "first\t0\t" << timed() << '\n';
    compare(f, "first");
    snapshot("first_inference");
    for (auto phase : {"warmup", "measured"}) {
      int count = std::string(phase) == "warmup" ? 20 : 100;
      std::vector<double> durations;
      for (int i = 0; i < count; ++i) durations.push_back(timed());
      const char* end = count == 20 ? "after_warmup" : "after_measurement";
      compare(f, end);
      snapshot(end);
      for (int i = 0; i < count; ++i)
        times << phase << '\t' << i << '\t' << durations[i] << '\n';
    }
    std::ifstream input("/proc/self/maps");
    require(bool(input), "Cannot read loaded libraries");
    std::string maps((std::istreambuf_iterator<char>(input)), {});
    std::ofstream mapped(directory / "loaded-maps.txt");
    mapped << maps;
    mapped.close();
    require(bool(mapped), "Cannot save loaded libraries");
    for (auto word : {"libtorch", "libpython", "libc10"})
      require(maps.find(word) == std::string::npos,
              "Unexpected native dependency: " + std::string(word));
    times.flush();
    require(bool(times), "Timing report failed");
    std::ofstream pass(directory / "PASS");
    pass << "All outputs passed at atol=rtol=3e-4\n";
    pass.close();
    require(bool(pass), "Cannot write PASS");
  }
};
}  // namespace pbr
