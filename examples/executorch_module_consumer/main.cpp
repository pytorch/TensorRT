// Runs a delegated program the way an application does: through ExecuTorch's module extension,
// linking the delegate out of its installed wheel rather than building it.
//
// Two shapes are covered, because they are not interchangeable. A program that plans its own
// outputs runs with forward() alone. A program exported to leave its outputs to the caller has no
// address to write to until one is supplied, so forward() by itself fails on it by design, and the
// caller has to hand a device buffer in first.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::from_blob;
using executorch::extension::make_tensor_ptr;
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
    std::fprintf(stderr, "usage: %s --model_path <file.pte> [--device_outputs] [--num_runs N]\n", argv[0]);
    return 2;
  }
  // Present means the program leaves its outputs to the caller, so one has to be supplied.
  bool caller_owns_outputs = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device_outputs") == 0) {
      caller_owns_outputs = true;
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
    const size_t bytes = count * sizeof(float);
    const std::vector<float> host(count, 1.0f);
    void* device = nullptr;
    if (!cuda_ok(cudaMalloc(&device, bytes), "cudaMalloc for an input")) {
      return 1;
    }
    owned.push_back(device);
    if (!cuda_ok(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice), "input upload")) {
      return 1;
    }
    inputs.push_back(from_blob(device, sizes, executorch::aten::ScalarType::Float));
    // forward() takes EValue, and an EValue borrows the tensor, so the pointers above are kept
    // alive in `inputs` for as long as the call needs them.
    input_values.emplace_back(*inputs.back());
  }

  std::vector<executorch::extension::TensorPtr> outputs_held;
  if (caller_owns_outputs) {
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
      void* device = nullptr;
      if (!cuda_ok(cudaMalloc(&device, count * sizeof(float)), "cudaMalloc for an output")) {
        return 1;
      }
      owned.push_back(device);
      auto out = from_blob(device, sizes, executorch::aten::ScalarType::Float);
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
          "--device_outputs so a buffer is supplied first.\n",
          run,
          static_cast<unsigned>(result.error()));
      return 1;
    }
    if (run + 1 == num_runs) {
      const auto& outputs = result.get();
      std::printf("ran %d time(s), %zu output(s)\n", num_runs, outputs.size());
      for (size_t o = 0; o < outputs.size(); ++o) {
        const auto tensor = outputs[o].toTensor();
        const size_t print_n = tensor.numel() < 4 ? static_cast<size_t>(tensor.numel()) : 4;
        std::vector<float> staged(print_n, 0.0f);
        cudaPointerAttributes attrs{};
        const void* src = tensor.const_data_ptr();
        if (cudaPointerGetAttributes(&attrs, src) == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
          if (!cuda_ok(
                  cudaMemcpy(staged.data(), src, print_n * sizeof(float), cudaMemcpyDeviceToHost), "output download")) {
            return 1;
          }
          std::printf("  output %zu lives on the device", o);
        } else {
          std::memcpy(staged.data(), src, print_n * sizeof(float));
          std::printf("  output %zu lives on the host", o);
        }
        for (size_t v = 0; v < print_n; ++v) {
          std::printf(" %g", staged[v]);
        }
        std::printf("\n");
      }
    }
  }

  for (void* p : owned) {
    cudaFree(p);
  }
  return 0;
}
