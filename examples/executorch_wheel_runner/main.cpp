/*
 * Runs a .pte against libraries that come only from installed packages.
 *
 * Two things make this more than a smoke test. It compares outputs against
 * expected values from a file, so a wrong result fails instead of merely being
 * printed. And with --gpu-inputs it allocates the input tensors in device memory
 * before handing them to the runtime, which is how an accelerator application
 * actually feeds data: the CUDA delegate moves memory device to device and never
 * stages through the host, so it expects tensors that are already resident.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#ifdef RUNNER_HAS_CUDA_DELEGATE
#include <cuda_runtime.h>
#endif

using executorch::extension::FileDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

const char* flag(int argc, char** argv, const char* name, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      return argv[i + 1];
    }
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) {
      return true;
    }
  }
  return false;
}

// Whitespace-separated floats, so a test can generate them from Python.
std::vector<float> read_floats(const char* path) {
  std::vector<float> values;
  FILE* file = std::fopen(path, "r");
  if (file == nullptr) {
    return values;
  }
  float value = 0.0f;
  while (std::fscanf(file, "%f", &value) == 1) {
    values.push_back(value);
  }
  std::fclose(file);
  return values;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const char* model_path = flag(argc, argv, "--model", "model.pte");
  const char* expected_path = flag(argc, argv, "--expected", nullptr);
  const double tolerance = std::atof(flag(argc, argv, "--tolerance", "1e-4"));
  const bool gpu_inputs = has_flag(argc, argv, "--gpu-inputs");

  if (gpu_inputs) {
#ifndef RUNNER_HAS_CUDA_DELEGATE
    std::fprintf(
        stderr, "--gpu-inputs needs a build with the CUDA delegate present\n");
    return 2;
#endif
  }

  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  if (!loader.ok()) {
    std::fprintf(stderr, "could not open '%s'\n", model_path);
    return 1;
  }

  Result<Program> program = Program::load(&loader.get());
  if (!program.ok()) {
    std::fprintf(stderr, "could not parse '%s'\n", model_path);
    return 1;
  }

  Result<const char*> method_name = program->get_method_name(0);
  if (!method_name.ok()) {
    std::fprintf(stderr, "the program exposes no methods\n");
    return 1;
  }

  Result<executorch::runtime::MethodMeta> meta =
      program->method_meta(*method_name);
  if (!meta.ok()) {
    std::fprintf(stderr, "could not read method metadata\n");
    return 1;
  }

  std::vector<uint8_t> method_arena(16u * 1024u * 1024u);
  std::vector<uint8_t> temp_arena(4u * 1024u * 1024u);
  MemoryAllocator method_allocator(
      static_cast<uint32_t>(method_arena.size()), method_arena.data());
  MemoryAllocator temp_allocator(
      static_cast<uint32_t>(temp_arena.size()), temp_arena.data());

  std::vector<std::vector<uint8_t>> planned;
  std::vector<Span<uint8_t>> planned_spans;
  for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
    const size_t size =
        static_cast<size_t>(meta->memory_planned_buffer_size(i).get());
    planned.emplace_back(size);
    planned_spans.push_back({planned.back().data(), size});
  }
  HierarchicalAllocator planned_memory(
      {planned_spans.data(), planned_spans.size()});
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method =
      program->load_method(*method_name, &memory_manager, nullptr);
  if (!method.ok()) {
    std::fprintf(
        stderr,
        "could not load method '%s': 0x%x\n",
        *method_name,
        static_cast<unsigned>(method.error()));
    return 1;
  }

  // Inputs are filled with ones so a reference can be computed for any model.
  std::vector<std::vector<float>> host_inputs;
  std::vector<std::vector<int32_t>> sizes;
  std::vector<std::vector<uint8_t>> dim_order;
  std::vector<std::vector<int32_t>> strides;
#ifdef RUNNER_HAS_CUDA_DELEGATE
  std::vector<void*> device_buffers;
#endif

  const size_t num_inputs = method->inputs_size();
  host_inputs.resize(num_inputs);
  sizes.resize(num_inputs);
  dim_order.resize(num_inputs);
  strides.resize(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    Result<executorch::runtime::TensorInfo> info = meta->input_tensor_meta(i);
    if (!info.ok()) {
      continue; // Not a tensor input; leave whatever the program defaults to.
    }
    // Float32 only, checked rather than assumed. The storage below is sized in
    // units of float, so a narrower type would get too few bytes and a wider one
    // would be misread. Supporting every type means dtype-aware allocation and
    // comparison, which is more than this example needs.
    if (info->scalar_type() != executorch::aten::ScalarType::Float) {
      std::fprintf(stderr,
                   "input %zu is not float32, which this runner does not "
                   "handle\n",
                   i);
      return 1;
    }
    const auto& shape = info->sizes();
    sizes[i].assign(shape.begin(), shape.end());
    dim_order[i].resize(shape.size());
    strides[i].resize(shape.size());
    int32_t stride = 1;
    for (size_t d = shape.size(); d-- > 0;) {
      dim_order[i][d] = static_cast<uint8_t>(d);
      strides[i][d] = stride;
      stride *= sizes[i][d];
    }
    host_inputs[i].assign(info->nbytes() / sizeof(float), 1.0f);

    void* data = host_inputs[i].data();
#ifdef RUNNER_HAS_CUDA_DELEGATE
    if (gpu_inputs) {
      // A memory-planned input is written into the plan's arena with a host
      // memcpy, so handing it a device pointer would make the host read device
      // memory. This runner allocates the arena on the host, so it takes the
      // aliasing path only for a non-planned input. Keeping activations on the
      // device instead means giving the runtime a device arena, which is a
      // different setup than this example builds.
      if (info->is_memory_planned()) {
        std::fprintf(
            stderr,
            "input %zu is memory planned and this runner allocates the plan on "
            "the host, so a device pointer would be copied by the host; run "
            "without --gpu-inputs for this program\n",
            i);
        return 1;
      }
      void* device = nullptr;
      if (cudaMalloc(&device, info->nbytes()) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed for input %zu\n", i);
        return 1;
      }
      if (cudaMemcpy(
              device, data, info->nbytes(), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
        std::fprintf(stderr, "staging input %zu to the device failed\n", i);
        return 1;
      }
      device_buffers.push_back(device);
      data = device;
    }
#endif

    auto* impl = method_allocator.allocateInstance<
        executorch::runtime::etensor::TensorImpl>();
    new (impl) executorch::runtime::etensor::TensorImpl(
        info->scalar_type(),
        static_cast<ssize_t>(sizes[i].size()),
        sizes[i].data(),
        data,
        dim_order[i].data(),
        strides[i].data());
    executorch::runtime::etensor::Tensor tensor(impl);
    const Error status = method->set_input(EValue(tensor), i);
    if (status != Error::Ok) {
      std::fprintf(
          stderr,
          "set_input(%zu) failed: 0x%x\n",
          i,
          static_cast<unsigned>(status));
      return 1;
    }
  }

  const Error status = method->execute();
  if (status != Error::Ok) {
    std::fprintf(
        stderr, "execute() failed: 0x%x\n", static_cast<unsigned>(status));
    return 1;
  }

  std::vector<EValue> outputs(method->outputs_size());
  if (method->get_outputs(outputs.data(), outputs.size()) != Error::Ok) {
    std::fprintf(stderr, "get_outputs() failed\n");
    return 1;
  }

  std::vector<float> produced;
  for (const EValue& value : outputs) {
    if (!value.isTensor()) {
      continue;
    }
    const auto tensor = value.toTensor();
    // Checked for the same reason as the inputs: the read below is in units of
    // float, so a narrower type would be read past its end and a wider one
    // misinterpreted.
    if (tensor.scalar_type() != executorch::aten::ScalarType::Float) {
      std::fprintf(stderr,
                   "an output is not float32, which this runner does not "
                   "handle\n");
      return 1;
    }
    const float* data = tensor.const_data_ptr<float>();
    if (data == nullptr) {
      std::fprintf(stderr, "an output tensor has no host-readable data\n");
      return 1;
    }
    produced.insert(produced.end(), data, data + tensor.numel());
  }

  std::printf("produced %zu output values\n", produced.size());
  for (size_t i = 0; i < produced.size() && i < 8; ++i) {
    std::printf("  [%zu] %.6f\n", i, produced[i]);
  }

#ifdef RUNNER_HAS_CUDA_DELEGATE
  for (void* buffer : device_buffers) {
    cudaFree(buffer);
  }
#endif

  if (expected_path == nullptr) {
    return 0;
  }

  const std::vector<float> expected = read_floats(expected_path);
  if (expected.empty()) {
    std::fprintf(stderr, "could not read expected values from a file\n");
    return 1;
  }
  if (expected.size() != produced.size()) {
    std::fprintf(
        stderr,
        "expected %zu values but the model produced %zu\n",
        expected.size(),
        produced.size());
    return 1;
  }
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double difference =
        std::abs(static_cast<double>(produced[i]) - expected[i]);
    if (difference > worst) {
      worst = difference;
    }
  }
  std::printf("largest difference from the reference: %g\n", worst);
  if (worst > tolerance) {
    std::fprintf(
        stderr, "outputs differ by %g, more than the %g allowed\n", worst,
        tolerance);
    return 1;
  }
  std::printf("outputs match the reference\n");
  return 0;
}
