/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Host-driven PI0.5 flow-matching loop for an Edge-LLM action-step .pte.
 *
 * The delegated TensorRT engine computes one velocity prediction:
 *
 *   velocity = action_step(x_t, timestep, prefix_k, prefix_v, positions, mask)
 *
 * This runner keeps the ExecuTorch Method and its delegate alive for the whole
 * rollout, then performs the Euler update on the host:
 *
 *   x_t = x_t + (-1 / num_steps) * velocity
 *
 * Prefix KV is zero-filled in this standalone smoke test. A full VLA runner
 * replaces those two inputs with prefix K/V produced by language prefill.
 *
 * Usage:
 *   edge_llm_action_loop_check \
 *       --model_path=/tmp/pi05_action_edge.pte \
 *       --num_steps=50
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

using executorch::extension::FileDataLoader;
using executorch::runtime::DeviceMemoryBuffer;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

namespace {

// ExecuTorch uses these allocators for method bookkeeping and temporary values.
// The large TensorRT tensors themselves live in memory-planned CPU/CUDA arenas.
constexpr size_t kMethodPoolBytes = 4 * 1024U * 1024U;
constexpr size_t kTempPoolBytes = 1 * 1024U * 1024U;

const char* get_flag(int argc, char** argv, const char* flag, const char* fallback) {
  const size_t length = std::strlen(flag);
  for (int index = 1; index < argc; ++index) {
    if (std::strncmp(argv[index], flag, length) == 0 && argv[index][length] == '=') {
      return argv[index] + length + 1;
    }
  }
  return fallback;
}

// Parse strictly so a typo cannot silently select zero iterations.
int parse_positive_int(const char* text, const char* flag) {
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  ET_CHECK_MSG(
      errno == 0 && end != text && *end == '\0' && value > 0 && value <= INT_MAX,
      "%s must be a positive integer, got '%s'",
      flag,
      text);
  return static_cast<int>(value);
}

float timestep_for_step(int step, int num_steps) {
  return 1.0F - static_cast<float>(step) / static_cast<float>(num_steps);
}

void euler_step(std::vector<float>& actions, const std::vector<float>& velocity, int num_steps) {
  ET_CHECK_MSG(actions.size() == velocity.size(), "Action and velocity sizes differ");
  const float dt = -1.0F / static_cast<float>(num_steps);
  for (size_t index = 0; index < actions.size(); ++index) {
    actions[index] += dt * velocity[index];
  }
}

// User inputs are host-backed. ExecuTorch's generated device-copy nodes move
// them into the CUDA planned arena before the Edge-LLM delegate executes.
void write_fp16(std::vector<uint8_t>& destination, const std::vector<float>& source) {
  ET_CHECK_MSG(
      destination.size() == source.size() * sizeof(__half), "FP16 input byte size does not match action count");
  for (size_t index = 0; index < source.size(); ++index) {
    const __half value = __float2half(source[index]);
    std::memcpy(destination.data() + index * sizeof(value), &value, sizeof(value));
  }
}

// Delegate outputs may be in a CUDA planned arena. cudaMemcpyDefault handles
// either host or device pointers and gives the host Euler loop float values.
std::vector<float> read_fp16(const void* source, size_t count) {
  std::vector<__half> fp16(count);
  ET_CHECK_MSG(
      cudaMemcpy(fp16.data(), source, count * sizeof(__half), cudaMemcpyDefault) == cudaSuccess,
      "Failed to copy action velocity");

  std::vector<float> result(count);
  for (size_t index = 0; index < count; ++index) {
    result[index] = __half2float(fp16[index]);
  }
  return result;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const char* model_path = get_flag(argc, argv, "--model_path", "pi05_action_edge.pte");
  const int num_steps = parse_positive_int(get_flag(argc, argv, "--num_steps", "50"), "--num_steps");

  // FileDataLoader must outlive Program because Program reads constants and
  // delegate payloads from it while loading the method.
  Result<FileDataLoader> loader_result = FileDataLoader::from(model_path);
  ET_CHECK_MSG(loader_result.ok(), "Failed to open '%s'", model_path);
  auto loader = std::make_unique<FileDataLoader>(std::move(loader_result.get()));

  Result<Program> program = Program::load(loader.get());
  ET_CHECK_MSG(program.ok(), "Failed to parse '%s'", model_path);
  auto method_name = program->get_method_name(0);
  ET_CHECK_MSG(method_name.ok(), "Program has no methods");

  Result<MethodMeta> method_meta = program->method_meta(*method_name);
  ET_CHECK_MSG(method_meta.ok(), "Failed to read action method metadata");
  ET_CHECK_MSG(method_meta->num_inputs() == 6, "PI0.5 action method must have six inputs");

  // Allocate ExecuTorch's small bookkeeping pools.
  auto method_pool = std::make_unique<uint8_t[]>(kMethodPoolBytes);
  auto temp_pool = std::make_unique<uint8_t[]>(kTempPoolBytes);
  MemoryAllocator method_allocator{kMethodPoolBytes, method_pool.get()};
  MemoryAllocator temp_allocator{kTempPoolBytes, temp_pool.get()};

  // Honor each memory-planned arena's device. Passing host memory for an arena
  // tagged CUDA would leave the delegate with a device-typed, host-backed tensor.
  std::vector<std::unique_ptr<uint8_t[]>> host_arenas;
  std::vector<DeviceMemoryBuffer> device_arenas;
  std::vector<Span<uint8_t>> planned_spans;
  for (size_t index = 0; index < method_meta->num_memory_planned_buffers(); ++index) {
    const size_t bytes = static_cast<size_t>(method_meta->memory_planned_buffer_size(index).get());
    auto device = method_meta->memory_planned_buffer_device(index);
    ET_CHECK_MSG(device.ok(), "Failed to read planned-buffer device %zu", index);

    if (device->is_cpu()) {
      host_arenas.push_back(std::make_unique<uint8_t[]>(bytes));
      planned_spans.push_back({host_arenas.back().get(), bytes});
    } else {
      Result<DeviceMemoryBuffer> arena = DeviceMemoryBuffer::create(bytes, device->type(), device->index());
      ET_CHECK_MSG(arena.ok(), "Failed to allocate device arena %zu", index);
      ET_CHECK_MSG(cudaMemset(arena->data(), 0, bytes) == cudaSuccess, "Failed to clear device arena %zu", index);
      planned_spans.push_back(arena->as_span());
      device_arenas.push_back(std::move(arena.get()));
    }
  }

  HierarchicalAllocator planned_memory{{planned_spans.data(), planned_spans.size()}};
  MemoryManager memory_manager{&method_allocator, &planned_memory, &temp_allocator};

  // Loading once creates one EdgeLLMHandle, action adapter, TensorRT engine, and
  // execution context. Every denoising step below reuses that same state.
  Result<Method> method = program->load_method(*method_name, &memory_manager, nullptr);
  ET_CHECK_MSG(method.ok(), "Failed to load action method");

  // Build persistent host-backed TensorImpls from the exported input metadata.
  // Their storage remains valid until after the final denoising iteration.
  const size_t num_inputs = method_meta->num_inputs();
  std::vector<std::vector<uint8_t>> input_data(num_inputs);
  std::vector<std::vector<exec_aten::SizesType>> input_sizes(num_inputs);
  std::vector<std::vector<exec_aten::DimOrderType>> input_dim_order(num_inputs);
  std::vector<std::vector<exec_aten::StridesType>> input_strides(num_inputs);
  std::vector<exec_aten::ScalarType> input_types(num_inputs);
  std::vector<exec_aten::TensorImpl> input_impls;
  input_impls.reserve(num_inputs);

  for (size_t index = 0; index < num_inputs; ++index) {
    Result<TensorInfo> info = method_meta->input_tensor_meta(index);
    ET_CHECK_MSG(info.ok(), "Failed to read input %zu metadata", index);
    input_types[index] = info->scalar_type();
    input_sizes[index].assign(info->sizes().begin(), info->sizes().end());
    input_data[index].assign(info->nbytes(), 0);

    const ssize_t dimensions = static_cast<ssize_t>(input_sizes[index].size());
    input_dim_order[index].resize(dimensions);
    input_strides[index].resize(dimensions);
    exec_aten::StridesType stride = 1;
    for (ssize_t dim = dimensions - 1; dim >= 0; --dim) {
      input_dim_order[index][dim] = static_cast<exec_aten::DimOrderType>(dim);
      input_strides[index][dim] = stride;
      stride *= input_sizes[index][dim];
    }

    input_impls.emplace_back(
        input_types[index],
        dimensions,
        input_sizes[index].data(),
        input_data[index].data(),
        input_dim_order[index].data(),
        input_strides[index].data());
  }

  // The exported PI0.5 binding order is:
  //   x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask.
  ET_CHECK_MSG(input_types[0] == exec_aten::ScalarType::Half, "x_t must be FP16");
  ET_CHECK_MSG(input_types[1] == exec_aten::ScalarType::Float, "timestep must be FP32");
  ET_CHECK_MSG(input_types[4] == exec_aten::ScalarType::Long, "position_ids must be int64");
  ET_CHECK_MSG(input_sizes[0].size() == 3, "x_t must have shape [B,H,D]");
  ET_CHECK_MSG(input_sizes[5].size() == 4, "attention_mask must have rank four");

  const int64_t batch_size = input_sizes[0][0];
  const int64_t chunk_size = input_sizes[0][1];
  const int64_t action_dim = input_sizes[0][2];
  const int64_t total_attention_length = input_sizes[5][3];
  const int64_t prefix_length = total_attention_length - chunk_size;
  ET_CHECK_MSG(batch_size > 0 && chunk_size > 0 && action_dim > 0, "Invalid action input shape");
  ET_CHECK_MSG(prefix_length >= 0, "Attention mask is shorter than the action suffix");

  const size_t action_count = static_cast<size_t>(batch_size * chunk_size * action_dim);
  std::vector<float> actions(action_count);
  for (size_t index = 0; index < action_count; ++index) {
    // Deterministic pseudo-noise makes repeated test runs reproducible.
    actions[index] = std::sin(static_cast<float>(index) * 0.01F);
  }

  // With an all-valid prefix and one shared diffusion block, the exported
  // additive attention mask is all zero. input_data was zero-initialized.
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    for (int64_t token = 0; token < chunk_size; ++token) {
      const int64_t position = prefix_length + token;
      const size_t offset = static_cast<size_t>(batch * chunk_size + token) * sizeof(position);
      std::memcpy(input_data[4].data() + offset, &position, sizeof(position));
    }
  }

  cudaStream_t caller_stream = nullptr;
  ET_CHECK_MSG(cudaStreamCreate(&caller_stream) == cudaSuccess, "Failed to create caller stream");

  {
    // Every delegate invocation observes and enqueues on this same stream.
    executorch::extension::cuda::CallerStreamGuard caller_stream_guard(caller_stream);
    for (int step = 0; step < num_steps; ++step) {
      write_fp16(input_data[0], actions);

      const float timestep = timestep_for_step(step, num_steps);
      for (int64_t batch = 0; batch < batch_size; ++batch) {
        std::memcpy(input_data[1].data() + static_cast<size_t>(batch) * sizeof(timestep), &timestep, sizeof(timestep));
      }

      // set_input copies each host tensor into the method's planned values.
      // Re-submit all six inputs because x_t and timestep change every step.
      for (size_t index = 0; index < num_inputs; ++index) {
        ET_CHECK_MSG(
            method->set_input(EValue(exec_aten::Tensor(&input_impls[index])), index) == Error::Ok,
            "Failed to set action input %zu at step %d",
            index,
            step);
      }

      ET_CHECK_MSG(method->execute() == Error::Ok, "Action execution failed at step %d", step);
      ET_CHECK_MSG(cudaStreamSynchronize(caller_stream) == cudaSuccess, "Action stream failed at step %d", step);

      EValue output;
      ET_CHECK_MSG(method->get_outputs(&output, 1) == Error::Ok, "Failed to read velocity at step %d", step);
      ET_CHECK_MSG(output.isTensor(), "Action output is not a tensor");
      exec_aten::Tensor velocity_tensor = output.toTensor();
      ET_CHECK_MSG(velocity_tensor.scalar_type() == exec_aten::ScalarType::Half, "Velocity must be FP16");
      ET_CHECK_MSG(static_cast<size_t>(velocity_tensor.numel()) == action_count, "Velocity shape does not match x_t");

      std::vector<float> velocity = read_fp16(velocity_tensor.const_data_ptr(), action_count);
      for (float value : velocity) {
        ET_CHECK_MSG(std::isfinite(value), "Velocity contains a non-finite value at step %d", step);
      }
      euler_step(actions, velocity, num_steps);
    }
  }

  ET_CHECK_MSG(cudaStreamDestroy(caller_stream) == cudaSuccess, "Failed to destroy caller stream");
  for (float value : actions) {
    ET_CHECK_MSG(std::isfinite(value), "Final actions contain a non-finite value");
  }

  std::fprintf(
      stderr, "[edge-llm-action-loop] PASS: ran %d steps with one persistent delegate; final first values:", num_steps);
  const size_t values_to_print = actions.size() < 8 ? actions.size() : 8;
  for (size_t index = 0; index < values_to_print; ++index) {
    std::fprintf(stderr, " %.5f", actions[index]);
  }
  std::fprintf(stderr, "\n");
  return 0;
}
