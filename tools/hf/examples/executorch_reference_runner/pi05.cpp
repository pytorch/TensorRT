// PI0.5 reference application: one vision/prefill, then a reused action method.
// Input files are written by torch_tensorrt_edge_llm.pi05.write_native_inputs.
// This correctness-oriented example stages prefix packing and Euler on the CPU.
#include <cuda_runtime.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace executorch::runtime;
using executorch::extension::FileDataLoader;

namespace {
const char* flag(int argc, char** argv, const char* name, const char* fallback) {
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], name, std::strlen(name)) == 0 && argv[i][std::strlen(name)] == '=') {
      return argv[i] + std::strlen(name) + 1;
    }
  }
  return fallback;
}

struct Input {
  std::vector<uint8_t> data;
  std::vector<exec_aten::SizesType> sizes;
  std::vector<exec_aten::DimOrderType> order;
  std::vector<exec_aten::StridesType> strides;
  std::unique_ptr<exec_aten::TensorImpl> impl;

  explicit Input(const TensorInfo& info) : data(info.nbytes()) {
    sizes.assign(info.sizes().begin(), info.sizes().end());
    order.resize(sizes.size());
    strides.resize(sizes.size());
    exec_aten::StridesType stride = 1;
    for (size_t d = sizes.size(); d-- > 0;) {
      order[d] = static_cast<exec_aten::DimOrderType>(d);
      strides[d] = stride;
      stride *= sizes[d];
    }
    impl = std::make_unique<exec_aten::TensorImpl>(
        info.scalar_type(), sizes.size(), sizes.data(), data.data(), order.data(), strides.data());
  }

  void read(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    ET_CHECK_MSG(stream.good() && stream.tellg() == static_cast<std::streamoff>(data.size()),
                 "Wrong input file size: %s (expected %zu)", path.c_str(), data.size());
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), data.size());
    ET_CHECK_MSG(stream.good(), "Could not read %s", path.c_str());
  }

  EValue value() { return EValue(exec_aten::Tensor(impl.get())); }
};

struct CudaDeleter {
  int device;
  void operator()(uint8_t* pointer) const {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    cudaFree(pointer);
    cudaSetDevice(previous);
  }
};

std::vector<uint8_t> host_copy(const exec_aten::Tensor& tensor) {
  std::vector<uint8_t> result(tensor.nbytes());
  if (tensor.device().is_cpu()) {
    std::memcpy(result.data(), tensor.const_data_ptr(), result.size());
  } else {
    auto status = cudaMemcpy(result.data(), tensor.const_data_ptr(), result.size(), cudaMemcpyDeviceToHost);
    ET_CHECK_MSG(status == cudaSuccess, "Output copy failed: %s", cudaGetErrorString(status));
  }
  return result;
}

// Each method retains its own allocator and planned memory for its lifetime.
// Methods are destroyed before the memory managers they reference.
struct Invocation {
  std::vector<uint8_t> method_pool = std::vector<uint8_t>(4 * 1024 * 1024);
  std::vector<uint8_t> temp_pool = std::vector<uint8_t>(1024 * 1024);
  MemoryAllocator method_allocator{static_cast<uint32_t>(method_pool.size()), method_pool.data()};
  MemoryAllocator temp_allocator{static_cast<uint32_t>(temp_pool.size()), temp_pool.data()};
  std::vector<std::unique_ptr<uint8_t[]>> cpu_buffers;
  std::vector<std::unique_ptr<uint8_t, CudaDeleter>> device_buffers;
  std::vector<Span<uint8_t>> spans;
  std::unique_ptr<HierarchicalAllocator> planned;
  std::unique_ptr<MemoryManager> memory;
  std::vector<std::unique_ptr<Input>> inputs;
  std::optional<Method> method;

  Invocation(Program& program, const char* name) {
    auto meta = program.method_meta(name);
    ET_CHECK_MSG(meta.ok(), "Missing method %s", name);
    for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
      auto size = meta->memory_planned_buffer_size(i);
      auto device = meta->memory_planned_buffer_device(i);
      ET_CHECK_MSG(size.ok() && device.ok(), "Invalid memory plan for %s", name);
      if (device->is_cpu()) {
        cpu_buffers.push_back(std::make_unique<uint8_t[]>(*size));
        spans.push_back({cpu_buffers.back().get(), static_cast<size_t>(*size)});
      } else {
        ET_CHECK_MSG(device->type() == etensor::DeviceType::CUDA, "Only CPU and CUDA memory are supported");
        int previous = 0;
        cudaGetDevice(&previous);
        ET_CHECK_MSG(cudaSetDevice(device->index()) == cudaSuccess, "Cannot select CUDA device");
        void* pointer = nullptr;
        ET_CHECK_MSG(cudaMalloc(&pointer, *size) == cudaSuccess, "Device allocation failed for %s", name);
        cudaSetDevice(previous);
        device_buffers.emplace_back(static_cast<uint8_t*>(pointer), CudaDeleter{device->index()});
        spans.push_back({static_cast<uint8_t*>(pointer), static_cast<size_t>(*size)});
      }
    }
    planned = std::make_unique<HierarchicalAllocator>(Span<Span<uint8_t>>(spans.data(), spans.size()));
    memory = std::make_unique<MemoryManager>(&method_allocator, planned.get(), &temp_allocator);
    auto loaded = program.load_method(name, memory.get());
    ET_CHECK_MSG(loaded.ok(), "Failed to initialize %s: %u", name, static_cast<unsigned>(loaded.error()));
    method.emplace(std::move(*loaded));
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      auto info = meta->input_tensor_meta(i);
      ET_CHECK_MSG(info.ok(), "Expected tensor input for %s", name);
      inputs.push_back(std::make_unique<Input>(*info));
    }
  }

  std::vector<EValue> run() {
    for (size_t i = 0; i < inputs.size(); ++i) {
      ET_CHECK_MSG(method->set_input(inputs[i]->value(), i) == Error::Ok, "set_input failed");
    }
    ET_CHECK_MSG(method->execute() == Error::Ok, "Method execution failed");
    std::vector<EValue> outputs(method->outputs_size());
    ET_CHECK_MSG(method->get_outputs(outputs.data(), outputs.size()) == Error::Ok, "get_outputs failed");
    return outputs;
  }
};

void require_float(const Input& input) {
  ET_CHECK_MSG(input.impl->scalar_type() == exec_aten::ScalarType::Float,
               "Reference host packing/Euler expects float32 inputs");
}
} // namespace

int main(int argc, char** argv) {
  runtime_init();
  const char* model_path = flag(argc, argv, "--model_path", "pi05.pte");
  const std::string directory = flag(argc, argv, "--inputs_dir", "inputs");
  const char* output_path = flag(argc, argv, "--output_path", "actions.bin");
  const int steps = std::atoi(flag(argc, argv, "--num_steps", "10"));
  ET_CHECK_MSG(steps > 0, "num_steps must be positive");
  auto loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(loader.ok(), "Cannot read program %s", model_path);
  auto program = Program::load(&*loader);
  ET_CHECK_MSG(program.ok(), "Cannot load program");
  Invocation vision(*program, "vision");
  Invocation prefill(*program, "prefill");
  Invocation action(*program, "action_step");
  ET_CHECK_MSG(vision.inputs.size() == 1 && prefill.inputs.size() == 3 && action.inputs.size() == 6,
               "Program does not follow the PI0.5 method contract");
  vision.inputs[0]->read(directory + "/pixels.bin");
  auto image_outputs = vision.run();
  auto image_tensor = image_outputs.at(0).toTensor();
  ET_CHECK_MSG(image_tensor.dim() == 3 && image_tensor.scalar_type() == exec_aten::ScalarType::Float,
               "Vision output must be float32 [B,C*S,H]");
  auto image = host_copy(image_tensor);
  auto& prefix = *prefill.inputs[0];
  require_float(prefix);
  ET_CHECK_MSG(prefix.sizes.size() == 3, "Prefix must be [B,S,H]");
  const size_t batch = prefix.sizes[0], sequence = prefix.sizes[1], hidden = prefix.sizes[2];
  ET_CHECK_MSG(static_cast<size_t>(image_tensor.size(0)) == batch &&
                   static_cast<size_t>(image_tensor.size(2)) == hidden, "Vision/prefix dimensions disagree");
  const size_t image_tokens = image_tensor.size(1);
  std::ifstream lang_file(directory + "/language_embeds.bin", std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(lang_file.good(), "Missing language_embeds.bin");
  const size_t lang_bytes = lang_file.tellg();
  ET_CHECK_MSG(lang_bytes % (batch * hidden * sizeof(float)) == 0, "Invalid language embeddings");
  const size_t lang_tokens = lang_bytes / (batch * hidden * sizeof(float));
  std::vector<uint8_t> language(lang_bytes);
  lang_file.seekg(0);
  lang_file.read(reinterpret_cast<char*>(language.data()), lang_bytes);
  ET_CHECK_MSG(lang_file.good(), "Cannot read language embeddings");
  std::vector<int64_t> indices(batch * sequence);
  std::ifstream index_file(directory + "/compact_index.bin", std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(index_file.good() && index_file.tellg() == static_cast<std::streamoff>(indices.size() * sizeof(int64_t)),
               "Invalid compact_index.bin");
  index_file.seekg(0);
  index_file.read(reinterpret_cast<char*>(indices.data()), indices.size() * sizeof(int64_t));
  ET_CHECK_MSG(index_file.good(), "Cannot read prefix indices");
  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < sequence; ++s) {
      const int64_t index = indices[b * sequence + s];
      ET_CHECK_MSG(index >= 0 && static_cast<size_t>(index) < image_tokens + lang_tokens, "Invalid prefix index");
      const bool is_image = static_cast<size_t>(index) < image_tokens;
      const size_t offset = is_image ? b * image_tokens + index : b * lang_tokens + index - image_tokens;
      const auto& source = is_image ? image : language;
      std::memcpy(prefix.data.data() + (b * sequence + s) * hidden * sizeof(float),
                  source.data() + offset * hidden * sizeof(float), hidden * sizeof(float));
    }
  }
  prefill.inputs[1]->read(directory + "/prefix_mask.bin");
  prefill.inputs[2]->read(directory + "/prefix_positions.bin");
  auto prefix_outputs = prefill.run();
  ET_CHECK_MSG(prefix_outputs.size() == 3, "Prefill must return hidden states and K/V");
  for (size_t i = 0; i < 2; ++i) {
    auto cache = host_copy(prefix_outputs[i + 1].toTensor());
    ET_CHECK_MSG(cache.size() == action.inputs[i + 2]->data.size(), "Prefix cache dimensions disagree");
    std::memcpy(action.inputs[i + 2]->data.data(), cache.data(), cache.size());
  }
  action.inputs[0]->read(directory + "/noise.bin");
  action.inputs[4]->read(directory + "/action_positions.bin");
  action.inputs[5]->read(directory + "/action_mask.bin");
  require_float(*action.inputs[0]);
  require_float(*action.inputs[1]);
  ET_CHECK_MSG(action.inputs[1]->data.size() == batch * sizeof(float), "Timestep must have one value per batch");
  for (int step = 0; step < steps; ++step) {
    const float time = 1.f - static_cast<float>(step) / steps;
    for (size_t b = 0; b < batch; ++b) {
      std::memcpy(action.inputs[1]->data.data() + b * sizeof(float), &time, sizeof(float));
    }
    auto outputs = action.run();
    auto velocity_tensor = outputs.at(0).toTensor();
    ET_CHECK_MSG(velocity_tensor.scalar_type() == exec_aten::ScalarType::Float &&
                     velocity_tensor.nbytes() == action.inputs[0]->data.size(), "Invalid velocity output");
    auto velocity = host_copy(velocity_tensor);
    auto& state = action.inputs[0]->data;
    for (size_t offset = 0; offset < state.size(); offset += sizeof(float)) {
      float x, v;
      std::memcpy(&x, state.data() + offset, sizeof(float));
      std::memcpy(&v, velocity.data() + offset, sizeof(float));
      x -= v / steps;
      std::memcpy(state.data() + offset, &x, sizeof(float));
    }
  }
  std::ofstream output(output_path, std::ios::binary);
  const auto& state = action.inputs[0]->data;
  output.write(reinterpret_cast<const char*>(state.data()), state.size());
  ET_CHECK_MSG(output.good(), "Cannot write actions");
  std::fprintf(stderr, "PI0.5 completed: %d steps, %zu action values\n", steps, state.size() / sizeof(float));
  return 0;
}
