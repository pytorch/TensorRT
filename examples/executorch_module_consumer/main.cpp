/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// Runs a delegated program the way an application does: through ExecuTorch's module extension,
// linking the delegate out of its installed wheel rather than building it.
//
// Two shapes are covered, because they are not interchangeable. A program that plans its own
// outputs runs with forward() alone. A program exported to leave its outputs to the caller has no
// address to write to until one is supplied, so forward() by itself fails on it by design, and the
// caller has to hand a device buffer in first.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {

const char* flag(int argc, char** argv, const char* name, const char* fallback) {
  const std::string want = std::string(name) + "=";
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], want.c_str(), want.size()) == 0) {
      return argv[i] + want.size();
    }
    if (std::strcmp(argv[i], name) == 0 && i + 1 < argc) {
      return argv[i + 1];
    }
  }
  return fallback;
}

// The value the export script recorded for this program, read from the file it writes beside the
// .pte. Returned as an optional so a missing file is a refusal rather than a silent pass: a gate
// that cannot find its reference has to say so, not accept whatever it was given.
std::optional<float> expected_value(const std::string& model_path) {
  std::ifstream file(model_path + ".expected");
  if (!file) {
    return std::nullopt;
  }
  std::string shape_line;
  float value = 0.0f;
  if (!std::getline(file, shape_line) || !(file >> value)) {
    return std::nullopt;
  }
  return value;
}

bool cuda_ok(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char** argv) {
  const char* const model_path = flag(argc, argv, "--model_path", nullptr);
  if (model_path == nullptr) {
    std::fprintf(stderr, "usage: %s --model_path <file.pte> [--device_io] [--num_runs N]\n", argv[0]);
    return 2;
  }
  // A program exported for device-resident boundaries wants device memory on both sides and leaves
  // its outputs to the caller. One exported for host boundaries wants host memory and plans its own
  // outputs, and handing it a device pointer crashes rather than failing, because something on the
  // host side reads through it.
  bool device_boundary = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device_io") == 0 || std::strcmp(argv[i], "--device_outputs") == 0) {
      device_boundary = true;
    }
  }
  const int num_runs = std::atoi(flag(argc, argv, "--num_runs", "1"));
  if (num_runs < 1) {
    std::fprintf(stderr, "--num_runs must be at least 1, got %d\n", num_runs);
    return 2;
  }

  Module module(model_path);
  if (module.load() != Error::Ok) {
    std::fprintf(stderr, "could not load %s\n", model_path);
    return 1;
  }
  const auto meta = module.method_meta("forward");
  if (!meta.ok()) {
    std::fprintf(stderr, "the program has no forward method\n");
    return 1;
  }

  // Inputs come from device memory, which is what the delegate binds without a staging copy.
  std::vector<void*> owned;
  std::deque<std::vector<uint8_t>> host_buffers;
  std::vector<executorch::extension::TensorPtr> inputs;
  std::vector<EValue> input_values;
  for (size_t i = 0; i < meta->num_inputs(); ++i) {
    const auto info = meta->input_tensor_meta(i);
    if (!info.ok()) {
      std::fprintf(stderr, "input %zu has no metadata\n", i);
      return 1;
    }
    std::vector<executorch::aten::SizesType> sizes(info->sizes().begin(), info->sizes().end());
    size_t count = 1;
    for (const auto extent : sizes) {
      count *= static_cast<size_t>(extent);
    }
    const auto type = info->scalar_type();
    const size_t bytes = count * executorch::runtime::elementSize(type);
    if (device_boundary) {
      void* device = nullptr;
      if (!cuda_ok(cudaMalloc(&device, bytes), "cudaMalloc for an input")) {
        return 1;
      }
      owned.push_back(device);
      // Ones, because that is what the export script feeds when it records the reference value
      // this run is checked against. Zeros would compare a different computation.
      const std::vector<float> filled(count, 1.0f);
      if (!cuda_ok(cudaMemcpy(device, filled.data(), bytes, cudaMemcpyHostToDevice), "input upload")) {
        return 1;
      }
      inputs.push_back(from_blob(device, sizes, type));
    } else {
      host_buffers.emplace_back(bytes, 0);
      std::fill_n(reinterpret_cast<float*>(host_buffers.back().data()), count, 1.0f);
      inputs.push_back(from_blob(host_buffers.back().data(), sizes, type));
    }
    // forward() takes EValue, and an EValue borrows the tensor, so the pointers above are kept
    // alive in `inputs` for as long as the call needs them.
    input_values.emplace_back(*inputs.back());
  }

  std::vector<executorch::extension::TensorPtr> outputs_held;
  if (device_boundary) {
    for (size_t o = 0; o < meta->num_outputs(); ++o) {
      const auto info = meta->output_tensor_meta(o);
      if (!info.ok()) {
        std::fprintf(stderr, "output %zu has no metadata\n", o);
        return 1;
      }
      std::vector<executorch::aten::SizesType> sizes(info->sizes().begin(), info->sizes().end());
      size_t count = 1;
      for (const auto extent : sizes) {
        count *= static_cast<size_t>(extent);
      }
      const auto type = info->scalar_type();
      void* device = nullptr;
      if (!cuda_ok(cudaMalloc(&device, count * executorch::runtime::elementSize(type)), "cudaMalloc for an output")) {
        return 1;
      }
      owned.push_back(device);
      auto out = from_blob(device, sizes, type);
      outputs_held.push_back(out);
      if (module.set_output(EValue(*out), o) != Error::Ok) {
        std::fprintf(stderr, "could not hand output %zu to the program\n", o);
        return 1;
      }
    }
  }

  for (int run = 0; run < num_runs; ++run) {
    const auto result = module.forward(input_values);
    if (!result.ok()) {
      std::fprintf(
          stderr,
          "run %d failed with status 0x%x. A program whose outputs it does not plan needs "
          "--device_io so a buffer is supplied first.\n",
          run,
          static_cast<unsigned>(result.error()));
      return 1;
    }
    if (run + 1 == num_runs) {
      const auto& outputs = result.get();
      std::printf("ran %d time(s), %zu output(s)\n", num_runs, outputs.size());
      // The value the export script recorded. Without it there is nothing to compare against, so
      // this refuses rather than printing numbers and returning success, which is what it used to
      // do: a run with every element wrong exited zero.
      const std::optional<float> want = expected_value(model_path);
      if (!want.has_value()) {
        std::fprintf(
            stderr,
            "no reference value beside %s, so the output cannot be checked. The export script "
            "writes it; run that first.\n",
            model_path);
        return 1;
      }
      for (size_t o = 0; o < outputs.size(); ++o) {
        const auto tensor = outputs[o].toTensor();
        if (tensor.scalar_type() != executorch::aten::ScalarType::Float) {
          std::fprintf(stderr, "output %zu is not float, so the recorded reference does not describe it\n", o);
          return 1;
        }
        const size_t count = static_cast<size_t>(tensor.numel());
        std::vector<float> host(count, 0.0f);
        cudaPointerAttributes attrs{};
        const void* src = tensor.const_data_ptr();
        const bool on_device =
            cudaPointerGetAttributes(&attrs, src) == cudaSuccess && attrs.type == cudaMemoryTypeDevice;
        cudaGetLastError();
        if (on_device) {
          if (!cuda_ok(
                  cudaMemcpy(host.data(), src, count * sizeof(float), cudaMemcpyDeviceToHost), "output download")) {
            return 1;
          }
        } else {
          std::memcpy(host.data(), src, count * sizeof(float));
        }
        // Where the program was asked for device boundaries, a host-backed output means the
        // arrangement did not take, and saying so is the entire point of asking for it.
        if (device_boundary && !on_device) {
          std::fprintf(stderr, "output %zu came back on the host in device mode\n", o);
          return 1;
        }
        // Every element, not the first four. A wrong tail is exactly what printing a prefix hides.
        size_t bad = 0;
        float worst = 0.0f;
        for (size_t v = 0; v < count; ++v) {
          if (!std::isfinite(host[v])) {
            std::fprintf(stderr, "output %zu element %zu is not finite\n", o, v);
            return 1;
          }
          const float error = std::fabs(host[v] - *want);
          if (error > worst) {
            worst = error;
          }
          if (error > 2e-3f) {
            ++bad;
          }
        }
        if (bad != 0) {
          std::fprintf(
              stderr,
              "output %zu: %zu of %zu elements differ from %g, worst %g\n",
              o,
              bad,
              count,
              static_cast<double>(*want),
              static_cast<double>(worst));
          return 1;
        }
        std::printf(
            "  output %zu: %zu elements match %g on the %s, worst %g\n",
            o,
            count,
            static_cast<double>(*want),
            on_device ? "device" : "host",
            static_cast<double>(worst));
      }
    }
  }

  for (void* p : owned) {
    cudaFree(p);
  }
  return 0;
}
