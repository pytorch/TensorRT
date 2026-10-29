// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Fixed-input, host-boundary FP32 benchmark. Not a camera or recurrent pipeline.
// Timer: Module::forward + device fence + materialization of owned CPU outputs.
// Fixture I/O, reference comparisons, memory snapshots and reports are untimed.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/runtime.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}
void cuda_check(cudaError_t status, const char* operation) {
  require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
size_t positive_count(const std::string& value) {
  require(
      !value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
      "Iteration counts must be positive integers");
  const auto count = std::stoull(value);
  require(count > 0 && count <= 100000, "Iteration count outside 1..100000");
  return count;
}
struct Stream {
  cudaStream_t value = nullptr;
  Stream() {
    cuda_check(cudaStreamCreate(&value), "cudaStreamCreate");
  }
  ~Stream() {
    if (value)
      cudaStreamDestroy(value);
  }
};
std::vector<float> read_floats(const fs::path& path, size_t count) {
  require(fs::file_size(path) == count * sizeof(float), "Wrong file size: " + path.string());
  std::vector<float> result(count);
  std::ifstream file(path, std::ios::binary);
  require(
      bool(file.read(reinterpret_cast<char*>(result.data()), count * sizeof(float))), "Cannot read " + path.string());
  require(
      std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); }),
      "Nonfinite fixture: " + path.string());
  return result;
}
std::vector<int64_t> read_shape(const fs::path& path) {
  std::ifstream file(path);
  require(bool(file), "Missing reference shape: " + path.string());
  std::vector<int64_t> shape;
  int64_t size;
  while (file >> size) {
    require(size > 0, "Only nonempty static tensors are supported");
    shape.push_back(size);
  }
  require(file.eof(), "Invalid reference shape: " + path.string());
  return shape;
}
long long read_kib(const fs::path& path, const std::string& key) {
  std::ifstream file(path);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    std::string found, unit;
    long long value;
    if (fields >> found >> value >> unit && found == key && unit == "kB")
      return value;
  }
  return -1; // Explicitly unavailable, never silently report zero.
}
void memory_snapshot(std::ostream& out, const char* phase) {
  size_t free = 0, total = 0;
  cuda_check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  out << phase << '\t' << read_kib("/proc/self/status", "VmRSS:") << '\t' << read_kib("/proc/self/status", "VmHWM:")
      << '\t' << read_kib("/proc/self/smaps_rollup", "Pss:") << '\t' << free << '\t' << total << '\n';
  require(bool(out), "Cannot write memory report");
}
struct Output {
  std::vector<int64_t> shape;
  std::vector<float> expected;
  std::vector<float> actual;
};
void materialize(const std::vector<EValue>& values, std::vector<Output>& outputs) {
  require(values.size() == outputs.size(), "Output count differs");
  for (size_t i = 0; i < outputs.size(); ++i) {
    require(values[i].isTensor(), "Non-tensor output unsupported");
    const auto tensor = values[i].toTensor();
    auto& output = outputs[i];
    require(tensor.scalar_type() == executorch::aten::ScalarType::Float, "Only FP32 outputs supported");
    require(static_cast<size_t>(tensor.dim()) == output.shape.size(), "Output rank differs");
    int64_t stride = 1;
    for (int64_t d = tensor.dim() - 1; d >= 0; --d) {
      require(tensor.size(d) == output.shape[d], "Output shape differs");
      require(tensor.size(d) <= 1 || tensor.strides()[d] == stride, "Output is not contiguous");
      stride *= tensor.size(d);
    }
    require(static_cast<size_t>(tensor.numel()) == output.actual.size(), "Output size differs");
    const auto* source = tensor.const_data_ptr<float>();
    cudaPointerAttributes attributes{};
    const auto status = cudaPointerGetAttributes(&attributes, source);
    if (status == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
      cuda_check(cudaMemcpy(output.actual.data(), source, tensor.nbytes(), cudaMemcpyDeviceToHost), "output download");
    } else {
      require(status == cudaSuccess || status == cudaErrorInvalidValue, "Unknown output memory type");
      cudaGetLastError();
      std::memcpy(output.actual.data(), source, tensor.nbytes());
    }
  }
}
void compare(const std::vector<Output>& outputs, std::ostream& report, const char* phase) {
  bool passed = true;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output = outputs[i];
    size_t bad = 0;
    double max_error = 0;
    for (size_t j = 0; j < output.actual.size(); ++j) {
      const double actual = output.actual[j], expected = output.expected[j];
      const double error = std::abs(actual - expected);
      if (!std::isfinite(actual) || error > 3e-4 + 3e-4 * std::abs(expected))
        ++bad;
      if (std::isfinite(error))
        max_error = std::max(max_error, error);
    }
    report << phase << '\t' << i << '\t' << output.actual.size() << '\t' << max_error << '\t' << bad << '\n';
    passed &= bad == 0;
  }
  report.flush();
  require(bool(report) && passed, "Numerical gate failed; timings are NOT valid benchmark results");
}
void save_maps(const fs::path& output) {
  std::ifstream file("/proc/self/maps");
  require(bool(file), "Cannot read process maps");
  const std::string maps((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::ofstream saved(output);
  saved << maps;
  require(bool(saved), "Cannot save process maps");
  for (const char* forbidden : {"libtorch", "libc10", "libpython", "/site-packages/", "/venvs/"}) {
    require(maps.find(forbidden) == std::string::npos, "Forbidden native dependency: " + std::string(forbidden));
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::string model, data, fixture, destination;
    size_t warmup = 20, iterations = 100;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg.rfind("--model=", 0) == 0)
        model = arg.substr(8);
      else if (arg.rfind("--data=", 0) == 0)
        data = arg.substr(7);
      else if (arg.rfind("--case=", 0) == 0)
        fixture = arg.substr(7);
      else if (arg.rfind("--output=", 0) == 0)
        destination = arg.substr(9);
      else if (arg.rfind("--warmup=", 0) == 0)
        warmup = positive_count(arg.substr(9));
      else if (arg.rfind("--iterations=", 0) == 0)
        iterations = positive_count(arg.substr(13));
      else
        throw std::runtime_error("Unknown argument: " + arg);
    }
    require(
        !model.empty() && !fixture.empty() && !destination.empty(),
        "Usage: benchmark_pte --model=file.pte [--data=file.ptd] --case=fixture "
        "--output=new-dir [--warmup=20] [--iterations=100]");
    require(fs::is_regular_file(model), "Missing model");
    require(data.empty() || fs::is_regular_file(data), "Missing external data");
    require(fs::is_directory(fixture), "Missing fixture directory");
    require(!fs::exists(destination), "Output directory already exists");
    require(fs::create_directories(destination), "Cannot create output directory");
    std::ofstream timing(fs::path(destination) / "latency.tsv");
    std::ofstream phases(fs::path(destination) / "phases.tsv");
    std::ofstream memory(fs::path(destination) / "memory.tsv");
    std::ofstream comparisons(fs::path(destination) / "comparisons.tsv");
    std::ofstream info(fs::path(destination) / "metadata.tsv");
    require(timing && phases && memory && comparisons && info, "Cannot open reports");
    timing << std::setprecision(12) << "phase\titeration\twall_ms\n";
    phases << std::setprecision(12) << "phase\twall_ms\n";
    comparisons << std::setprecision(12) << "phase\toutput\telements\tmax_abs_error\tmismatches\n";
    memory << "phase\trss_kib\tprocess_hwm_kib\tpss_kib\tcuda_free_bytes\tcuda_total_bytes\n";
    info << "key\tvalue\nmodel\t" << model << "\nexternal_data\t" << data << "\nfixture\t" << fixture << "\nwarmup\t"
         << warmup << "\niterations\t" << iterations
         << "\nboundary\tforward + device fence + owned CPU output materialization"
         << "\ninput_mode\tfixed snapshot, no recurrent feedback\n";

    auto start = Clock::now();
    executorch::runtime::runtime_init();
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    Stream stream;
    cuda_check(cudaDeviceSynchronize(), "CUDA initialization fence");
    phases << "runtime_cuda_init\t" << elapsed(start) << '\n';
    executorch::extension::cuda::CallerStreamGuard guard(stream.value);
    memory_snapshot(memory, "cuda_initialized");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    info << "gpu\t" << prop.name << "\ncompute_capability\t" << prop.major << '.' << prop.minor << '\n';

    start = Clock::now();
    auto module = std::make_unique<Module>(model, data);
    require(module->load() == Error::Ok, "Module.load failed");
    phases << "module_load\t" << elapsed(start) << '\n';
    start = Clock::now();
    require(module->load_method("forward") == Error::Ok, "load_method failed");
    cuda_check(cudaDeviceSynchronize(), "method load fence");
    phases << "method_load\t" << elapsed(start) << '\n';
    memory_snapshot(memory, "method_loaded");
    const auto meta = module->method_meta("forward");
    require(meta.ok(), "Missing forward metadata");
    info << "TensorRTBackend\t" << meta->uses_backend("TensorRTBackend") << "\nCudaBackend\t"
         << meta->uses_backend("CudaBackend") << '\n';

    start = Clock::now();
    std::vector<std::vector<float>> inputs(meta->num_inputs());
    std::vector<TensorPtr> tensors;
    std::vector<EValue> values;
    size_t input_bytes = 0, output_bytes = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto desc = meta->input_tensor_meta(i);
      require(
          desc.ok() && desc->scalar_type() == executorch::aten::ScalarType::Float, "Only FP32 tensor inputs supported");
      for (size_t d = 0; d < desc->dim_order().size(); ++d) {
        require(desc->dim_order()[d] == d, "Only contiguous inputs supported");
      }
      inputs[i] =
          read_floats(fs::path(fixture) / ("input_" + std::to_string(i) + ".bin"), desc->nbytes() / sizeof(float));
      input_bytes += desc->nbytes();
      std::vector<executorch::aten::SizesType> shape(desc->sizes().begin(), desc->sizes().end());
      tensors.push_back(executorch::extension::from_blob(inputs[i].data(), shape, desc->scalar_type()));
      values.emplace_back(*tensors.back());
    }
    std::vector<Output> outputs(meta->num_outputs());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto desc = meta->output_tensor_meta(i);
      require(
          desc.ok() && desc->scalar_type() == executorch::aten::ScalarType::Float,
          "Only FP32 tensor outputs supported");
      const auto stem = fs::path(fixture) / ("output_" + std::to_string(i));
      outputs[i].shape = read_shape(stem.string() + ".shape");
      outputs[i].expected = read_floats(stem.string() + ".bin", desc->nbytes() / sizeof(float));
      outputs[i].actual.resize(outputs[i].expected.size());
      output_bytes += desc->nbytes();
    }
    phases << "fixture_load\t" << elapsed(start) << '\n';
    info << "input_bytes\t" << input_bytes << "\noutput_bytes\t" << output_bytes << '\n';
    memory_snapshot(memory, "fixtures_loaded");

    auto run = [&]() {
      const auto begin = Clock::now();
      const auto result = module->forward(values);
      require(result.ok(), "forward failed");
      cuda_check(cudaDeviceSynchronize(), "inference fence");
      materialize(*result, outputs);
      return elapsed(begin);
    };
    const double first_ms = run();
    timing << "first\t0\t" << first_ms << '\n';
    compare(outputs, comparisons, "first");
    memory_snapshot(memory, "first_inference");

    // No file writes inside the timed loop, and no output comparison between samples.
    std::vector<double> warmup_times, measured_times;
    warmup_times.reserve(warmup);
    measured_times.reserve(iterations);
    for (size_t i = 0; i < warmup; ++i)
      warmup_times.push_back(run());
    compare(outputs, comparisons, "after_warmup");
    memory_snapshot(memory, "after_warmup");
    for (size_t i = 0; i < iterations; ++i)
      measured_times.push_back(run());
    compare(outputs, comparisons, "after_measurement");
    memory_snapshot(memory, "after_measurement");
    save_maps(fs::path(destination) / "loaded-maps.txt");
    for (size_t i = 0; i < warmup_times.size(); ++i)
      timing << "warmup\t" << i << '\t' << warmup_times[i] << '\n';
    for (size_t i = 0; i < measured_times.size(); ++i)
      timing << "measured\t" << i << '\t' << measured_times[i] << '\n';
    timing.flush();
    phases.flush();
    memory.flush();
    info.flush();
    require(timing && phases && memory && info, "Cannot flush reports");
    std::ofstream success(fs::path(destination) / "PASS");
    success << "All-output comparisons passed before/after timing; no Torch/Python mapped.\n";
    success.close();
    require(bool(success), "Cannot write PASS marker");
    std::cout << "PASS: warmup=" << warmup << " measured=" << iterations << " first_ms=" << first_ms
              << " fixed-input host-boundary benchmark\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
