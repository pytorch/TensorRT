// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Native correctness gate for static, contiguous FP32, host-boundary .pte files.
// Inputs/reference outputs are raw little-endian float files, prepared offline.
// This is a validation tool, not a benchmark or a camera preprocessing pipeline.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const char* operation) {
  require(status == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(status));
}

std::vector<float> read_floats(const fs::path& path, size_t count) {
  require(fs::file_size(path) == count * sizeof(float), "Wrong file size: " + path.string());
  std::vector<float> values(count);
  std::ifstream file(path, std::ios::binary);
  require(bool(file.read(reinterpret_cast<char*>(values.data()), count * sizeof(float))),
          "Cannot read " + path.string());
  return values;
}

void check_shape(const fs::path& path, const executorch::aten::Tensor& tensor) {
  std::ifstream file(path);
  require(bool(file), "Missing reference shape: " + path.string());
  std::vector<int64_t> shape;
  int64_t size;
  while (file >> size) shape.push_back(size);
  require(file.eof(), "Invalid reference shape: " + path.string());
  require(shape.size() == static_cast<size_t>(tensor.dim()), "Output rank differs");
  for (size_t d = 0; d < shape.size(); ++d) {
    require(shape[d] == tensor.size(d), "Output shape differs at dimension " + std::to_string(d));
  }
}

// Own the stream until after the module and its device allocations are destroyed.
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { check_cuda(cudaStreamCreate(&value), "cudaStreamCreate"); }
  ~Stream() { if (value) cudaStreamDestroy(value); }
};

struct Frame {
  fs::path directory;
  bool reset = false;
};

// A sequence manifest uses two whitespace-separated fields per non-comment line:
// reset|continue fixture-directory-relative-to-the-manifest. No Python is used.
std::vector<Frame> read_sequence(const fs::path& manifest) {
  std::ifstream file(manifest);
  require(bool(file), "Cannot read sequence manifest: " + manifest.string());
  std::vector<Frame> frames;
  std::string line;
  while (std::getline(file, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos ||
        line[line.find_first_not_of(" \t\r")] == '#') continue;
    std::istringstream fields(line);
    std::string mode, directory, extra;
    require(bool(fields >> mode >> directory) && !(fields >> extra),
            "Expected reset|continue and a fixture directory in: " + line);
    require(mode == "reset" || mode == "continue", "Invalid sequence mode: " + mode);
    require(!fs::path(directory).is_absolute(), "Sequence fixture paths must be relative");
    const auto path = manifest.parent_path() / directory;
    require(fs::is_directory(path), "Fixture directory missing: " + path.string());
    frames.push_back({path, mode == "reset"});
  }
  require(!frames.empty() && frames.front().reset, "Sequence must begin with a reset frame");
  return frames;
}
} // namespace

int main(int argc, char** argv) {
  try {
    std::string model, data, case_dir, output_dir, feedback, sequence;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg.rfind("--model=", 0) == 0) model = arg.substr(8);
      else if (arg.rfind("--data=", 0) == 0) data = arg.substr(7);
      else if (arg.rfind("--case=", 0) == 0) case_dir = arg.substr(7);
      else if (arg.rfind("--output=", 0) == 0) output_dir = arg.substr(9);
      // Explicit BEVFormer recurrence check: previous run's next_bev -> input 3.
      else if (arg.rfind("--bev-feedback=", 0) == 0) feedback = arg.substr(15);
      else if (arg.rfind("--bev-sequence=", 0) == 0) sequence = arg.substr(15);
      else throw std::runtime_error("Unknown argument: " + arg);
    }
    require(!model.empty() && !output_dir.empty() &&
            (case_dir.empty() != sequence.empty()) && (sequence.empty() || feedback.empty()),
      "Usage: validate_pte --model=file.pte [--data=file.ptd] --output=new-dir "
      "(--case=dir [--bev-feedback=previous-output_2.bin] | --bev-sequence=manifest.txt)");
    require(!fs::exists(output_dir), "Output directory already exists; use a new directory");
    require(fs::is_regular_file(model), "Model file missing");
    require(data.empty() || fs::is_regular_file(data), "External data file missing");
    const auto frames = sequence.empty() ? std::vector<Frame>{{case_dir, false}}
                                         : read_sequence(sequence);
    executorch::runtime::runtime_init();
    Stream stream;
    executorch::extension::cuda::CallerStreamGuard guard(stream.value);
    Module module(model, data);
    require(module.load() == Error::Ok, "Module.load failed");
    const auto meta = module.method_meta("forward");
    require(meta.ok(), "forward metadata missing");
    std::cout << "backends: TensorRT=" << meta->uses_backend("TensorRTBackend")
              << " CUDA=" << meta->uses_backend("CudaBackend") << '\n';

    const size_t ninputs = meta->num_inputs();
    if (!sequence.empty()) {
      require(ninputs == 5 && meta->num_outputs() == 3,
              "BEV sequence mode requires five inputs and three outputs");
      require(meta->uses_backend("TensorRTBackend") && meta->uses_backend("CudaBackend"),
              "This BEV sequence check expects both TensorRT and CUDA delegates");
    }
    fs::create_directories(output_dir);
    std::ofstream report(fs::path(output_dir) / "comparisons.tsv");
    require(bool(report), "Cannot create comparison report");
    report << "frame\tmode\toutput\telements\tmax_abs_error\tmismatches\n";
    bool passed = true;
    size_t total_elements = 0, resets = 0;
    std::vector<float> previous_bev;
    // Module and caller stream remain alive across all frames. Only the inputs
    // and returned views are recreated. Keep our own copy of recurrent state:
    // the next forward may reuse the runtime's output memory.
    for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
      const auto& frame = frames[frame_index];
      case_dir = frame.directory.string();
      const fs::path frame_output = sequence.empty() ? fs::path(output_dir)
          : fs::path(output_dir) / ("frame_" + std::to_string(frame_index));
      fs::create_directories(frame_output);
      std::cout << "frame=" << frame_index << " mode="
                << (sequence.empty() ? "single" : frame.reset ? "reset" : "continue")
                << " fixture=" << frame.directory << '\n';
      if (frame.reset) { previous_bev.clear(); ++resets; }
      std::vector<std::vector<float>> storage(ninputs);
      std::vector<TensorPtr> tensors;
      std::vector<EValue> values;
      for (size_t i = 0; i < ninputs; ++i) {
        const auto info = meta->input_tensor_meta(i);
        require(info.ok(), "Non-tensor input unsupported");
        require(info->scalar_type() == executorch::aten::ScalarType::Float, "Only FP32 inputs supported");
        for (size_t d = 0; d < info->dim_order().size(); ++d) {
          require(info->dim_order()[d] == d, "Only contiguous input layout supported");
        }
        const fs::path input = (i == 3 && !feedback.empty()) ? fs::path(feedback)
            : fs::path(case_dir) / ("input_" + std::to_string(i) + ".bin");
        const size_t count = info->nbytes() / sizeof(float);
        if (!sequence.empty() && i == 3) {
          if (frame.reset) storage[i].assign(count, 0.0f);
          else {
            require(previous_bev.size() == count, "Native BEV state missing or wrong size");
            storage[i] = previous_bev;
          }
        } else storage[i] = read_floats(input, count);
        if (!sequence.empty() && i == 4) {
          require(storage[i].size() == 1 && storage[i][0] == (frame.reset ? 0.0f : 1.0f),
                  "Fixture has_previous flag disagrees with sequence reset/continue mode");
        }
        std::vector<executorch::aten::SizesType> shape(info->sizes().begin(), info->sizes().end());
        tensors.push_back(executorch::extension::from_blob(storage[i].data(), shape, info->scalar_type()));
        values.emplace_back(*tensors.back());
      }

      const auto result = module.forward(values);
      require(result.ok(), "Module.forward failed with error " + std::to_string(static_cast<int>(result.error())));
      // Correctness fence, deliberately not a performance measurement.
      check_cuda(cudaDeviceSynchronize(), "inference synchronization");
      require(result->size() == meta->num_outputs(), "Output count differs from metadata");
      constexpr double atol = 3e-4, rtol = 3e-4;
      for (size_t i = 0; i < result->size(); ++i) {
        require(result->at(i).isTensor(), "Non-tensor output unsupported");
        const auto tensor = result->at(i).toTensor();
        require(tensor.scalar_type() == executorch::aten::ScalarType::Float, "Only FP32 outputs supported");
        const std::string stem = "output_" + std::to_string(i);
        check_shape(fs::path(case_dir) / (stem + ".shape"), tensor);
        // Fail rather than interpret a non-contiguous tensor as a flat array.
        int64_t stride = 1;
        for (int64_t d = tensor.dim() - 1; d >= 0; --d) {
          require(tensor.size(d) <= 1 || tensor.strides()[d] == stride, "Output is not contiguous");
          stride *= tensor.size(d);
        }
        std::vector<float> actual(tensor.numel());
        const auto expected = read_floats(fs::path(case_dir) / (stem + ".bin"), actual.size());
        const float* source = tensor.const_data_ptr<float>();
        cudaPointerAttributes attributes{};
        const auto queried = cudaPointerGetAttributes(&attributes, source);
        if (queried == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
          check_cuda(cudaMemcpy(actual.data(), source, tensor.nbytes(), cudaMemcpyDeviceToHost), "output download");
        } else {
          require(queried == cudaSuccess || queried == cudaErrorInvalidValue, "Cannot determine output memory type");
          cudaGetLastError();
          std::memcpy(actual.data(), source, tensor.nbytes());
        }
        size_t mismatches = 0;
        double max_error = 0;
        for (size_t j = 0; j < actual.size(); ++j) {
          const double diff = std::abs(double(actual[j]) - double(expected[j]));
          const bool finite = std::isfinite(actual[j]) && std::isfinite(expected[j]);
          if (!finite || diff > atol + rtol * std::abs(double(expected[j]))) ++mismatches;
          if (finite) max_error = std::max(max_error, diff);
        }
        std::cout << stem << " elements=" << actual.size() << " max_abs_error=" << max_error
                  << " mismatches=" << mismatches << '\n';
        passed &= mismatches == 0;
        total_elements += actual.size();
        report << frame_index << '\t' << (sequence.empty() ? "single" : frame.reset ? "reset" : "continue")
               << '\t' << i << '\t' << actual.size() << '\t' << max_error << '\t' << mismatches << '\n';
        if (!sequence.empty() && i == 2) previous_bev = actual;
        std::ofstream output(frame_output / (stem + ".bin"), std::ios::binary);
        require(bool(output.write(reinterpret_cast<const char*>(actual.data()), tensor.nbytes())), "Output write failed");
      }
      // Captures dlopen'ed AOTI modules too, while the Module is still alive.
      std::ifstream maps("/proc/self/maps");
      std::ofstream saved_maps(frame_output / "loaded-maps.txt");
      require(bool(maps) && bool(saved_maps), "Cannot capture loaded library maps");
      saved_maps << maps.rdbuf();
      require(bool(saved_maps), "Cannot write library maps");
    } // frames
    require(bool(report), "Cannot write comparison report");
    std::cout << "modules_loaded=1 frames=" << frames.size() << " scene_resets=" << resets
              << " compared_elements=" << total_elements << '\n';
    std::cout << (passed ? "PASS" : "FAIL") << ": all-output native comparison, rtol=atol=3e-4\n";
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
