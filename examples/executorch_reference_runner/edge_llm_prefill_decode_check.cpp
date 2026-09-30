/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/device_memory_buffer.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

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

struct HostTensor {
  exec_aten::ScalarType dtype;
  std::vector<int32_t> sizes;
  std::vector<uint8_t> data;
};

const char* get_flag(int argc, char** argv, const char* flag, const char* fallback) {
  const size_t length = std::strlen(flag);
  for (int index = 1; index < argc; ++index) {
    if (std::strncmp(argv[index], flag, length) == 0 && argv[index][length] == '=') {
      return argv[index] + length + 1;
    }
  }
  return fallback;
}

size_t element_size(exec_aten::ScalarType dtype) {
  switch (dtype) {
    case exec_aten::ScalarType::Half:
    case exec_aten::ScalarType::BFloat16:
      return 2;
    case exec_aten::ScalarType::Float:
    case exec_aten::ScalarType::Int:
      return 4;
    case exec_aten::ScalarType::Long:
      return 8;
    default:
      ET_CHECK_MSG(false, "unsupported scalar type %d", static_cast<int>(dtype));
      return 0;
  }
}

void write_int32(HostTensor& tensor, int32_t value) {
  ET_CHECK_MSG(tensor.dtype == exec_aten::ScalarType::Int && tensor.data.size() >= sizeof(value), "expected int32");
  std::memcpy(tensor.data.data(), &value, sizeof(value));
}

void write_int64(HostTensor& tensor, int64_t value) {
  ET_CHECK_MSG(tensor.dtype == exec_aten::ScalarType::Long && tensor.data.size() >= sizeof(value), "expected int64");
  std::memcpy(tensor.data.data(), &value, sizeof(value));
}

void populate_decode_inputs(std::vector<HostTensor>& inputs, const std::vector<HostTensor>& prefill_outputs) {
  ET_CHECK_MSG(inputs.size() >= 7, "decode program has too few inputs");
  ET_CHECK_MSG(prefill_outputs.size() == 4, "prefill program must return four outputs");
  const HostTensor& prefix_k = prefill_outputs[2];
  const HostTensor& prefix_v = prefill_outputs[3];
  ET_CHECK_MSG(prefix_k.dtype == exec_aten::ScalarType::Half && prefix_v.dtype == prefix_k.dtype, "expected FP16 KV");
  ET_CHECK_MSG(prefix_k.sizes == prefix_v.sizes && prefix_k.sizes.size() == 5, "invalid prefix KV shapes");

  const int64_t layers = prefix_k.sizes[0];
  const int64_t batch = prefix_k.sizes[1];
  const int64_t heads = prefix_k.sizes[2];
  const int64_t prefix_length = prefix_k.sizes[3];
  const int64_t head_size = prefix_k.sizes[4];
  ET_CHECK_MSG(inputs.size() == static_cast<size_t>(6 + layers), "decode KV input count does not match layer count");

  write_int32(inputs[2], static_cast<int32_t>(prefix_length + 1));
  write_int32(inputs[3], static_cast<int32_t>(prefix_length));
  write_int64(inputs[4], 0);

  const size_t scalar_bytes = element_size(prefix_k.dtype);
  const size_t token_row_bytes = static_cast<size_t>(prefix_length * head_size) * scalar_bytes;
  for (int64_t layer = 0; layer < layers; ++layer) {
    HostTensor& cache = inputs[static_cast<size_t>(6 + layer)];
    ET_CHECK_MSG(cache.dtype == prefix_k.dtype && cache.sizes.size() == 5, "invalid decode KV input");
    ET_CHECK_MSG(
        cache.sizes[0] == batch && cache.sizes[1] == 2 && cache.sizes[2] == heads && cache.sizes[4] == head_size &&
            cache.sizes[3] >= prefix_length,
        "decode KV capacity does not match prefill output");
    const int64_t capacity = cache.sizes[3];

    for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
      for (int64_t head = 0; head < heads; ++head) {
        const size_t source_offset =
            static_cast<size_t>((((layer * batch + batch_index) * heads + head) * prefix_length) * head_size) *
            scalar_bytes;
        for (int64_t kv = 0; kv < 2; ++kv) {
          const HostTensor& source = kv == 0 ? prefix_k : prefix_v;
          const size_t destination_offset =
              static_cast<size_t>((((batch_index * 2 + kv) * heads + head) * capacity) * head_size) * scalar_bytes;
          std::memcpy(cache.data.data() + destination_offset, source.data.data() + source_offset, token_row_bytes);
        }
      }
    }
  }
}

std::vector<HostTensor> run_program(
    const char* model_path,
    const std::vector<HostTensor>* prefill_outputs,
    cudaStream_t stream) {
  Result<FileDataLoader> loader_result = FileDataLoader::from(model_path);
  ET_CHECK_MSG(loader_result.ok(), "failed to open '%s'", model_path);
  auto loader = std::make_unique<FileDataLoader>(std::move(loader_result.get()));
  Result<Program> program = Program::load(loader.get());
  ET_CHECK_MSG(program.ok(), "failed to parse '%s'", model_path);
  auto method_name = program->get_method_name(0);
  ET_CHECK_MSG(method_name.ok(), "program has no methods");
  Result<MethodMeta> meta = program->method_meta(*method_name);
  ET_CHECK_MSG(meta.ok(), "failed to read method metadata");

  auto method_pool = std::make_unique<uint8_t[]>(4 * 1024U * 1024U);
  auto temp_pool = std::make_unique<uint8_t[]>(1 * 1024U * 1024U);
  MemoryAllocator method_allocator{4 * 1024U * 1024U, method_pool.get()};
  MemoryAllocator temp_allocator{1 * 1024U * 1024U, temp_pool.get()};

  std::vector<std::unique_ptr<uint8_t[]>> host_arenas;
  std::vector<DeviceMemoryBuffer> device_arenas;
  std::vector<Span<uint8_t>> planned_spans;
  for (size_t index = 0; index < meta->num_memory_planned_buffers(); ++index) {
    const size_t bytes = static_cast<size_t>(meta->memory_planned_buffer_size(index).get());
    auto device = meta->memory_planned_buffer_device(index);
    ET_CHECK_MSG(device.ok(), "failed to read planned-buffer device");
    if (device->is_cpu()) {
      host_arenas.push_back(std::make_unique<uint8_t[]>(bytes));
      planned_spans.push_back({host_arenas.back().get(), bytes});
    } else {
      Result<DeviceMemoryBuffer> arena = DeviceMemoryBuffer::create(bytes, device->type(), device->index());
      ET_CHECK_MSG(arena.ok(), "failed to allocate device arena");
      ET_CHECK_MSG(cudaMemset(arena->data(), 0, bytes) == cudaSuccess, "failed to clear device arena");
      planned_spans.push_back(arena->as_span());
      device_arenas.push_back(std::move(arena.get()));
    }
  }
  HierarchicalAllocator planned_memory{{planned_spans.data(), planned_spans.size()}};
  MemoryManager memory_manager{&method_allocator, &planned_memory, &temp_allocator};
  Result<Method> method = program->load_method(*method_name, &memory_manager, nullptr);
  ET_CHECK_MSG(method.ok(), "failed to load method from '%s'", model_path);

  std::vector<HostTensor> input_data;
  std::vector<std::vector<exec_aten::DimOrderType>> dim_orders;
  std::vector<std::vector<exec_aten::StridesType>> strides;
  std::vector<exec_aten::TensorImpl> input_impls;
  input_data.reserve(meta->num_inputs());
  dim_orders.resize(meta->num_inputs());
  strides.resize(meta->num_inputs());
  input_impls.reserve(meta->num_inputs());
  for (size_t index = 0; index < meta->num_inputs(); ++index) {
    Result<TensorInfo> tensor = meta->input_tensor_meta(index);
    ET_CHECK_MSG(tensor.ok(), "failed to read input %zu", index);
    HostTensor host{tensor->scalar_type(), {}, std::vector<uint8_t>(tensor->nbytes(), 0)};
    host.sizes.assign(tensor->sizes().begin(), tensor->sizes().end());
    input_data.push_back(std::move(host));
  }
  if (prefill_outputs != nullptr) {
    populate_decode_inputs(input_data, *prefill_outputs);
  }
  for (size_t index = 0; index < input_data.size(); ++index) {
    const ssize_t dimensions = static_cast<ssize_t>(input_data[index].sizes.size());
    dim_orders[index].resize(dimensions);
    strides[index].resize(dimensions);
    exec_aten::StridesType stride = 1;
    for (ssize_t dim = dimensions - 1; dim >= 0; --dim) {
      dim_orders[index][dim] = static_cast<exec_aten::DimOrderType>(dim);
      strides[index][dim] = stride;
      stride *= input_data[index].sizes[dim];
    }
    input_impls.emplace_back(
        input_data[index].dtype,
        dimensions,
        input_data[index].sizes.data(),
        input_data[index].data.data(),
        dim_orders[index].data(),
        strides[index].data());
    ET_CHECK_MSG(
        method->set_input(EValue(exec_aten::Tensor(&input_impls.back())), index) == Error::Ok,
        "failed to set input %zu",
        index);
  }

  {
    executorch::extension::cuda::CallerStreamGuard stream_guard(stream);
    ET_CHECK_MSG(method->execute() == Error::Ok, "execution failed for '%s'", model_path);
  }
  ET_CHECK_MSG(cudaStreamSynchronize(stream) == cudaSuccess, "caller stream synchronization failed");

  std::vector<EValue> output_values(method->outputs_size());
  ET_CHECK_MSG(method->get_outputs(output_values.data(), output_values.size()) == Error::Ok, "failed to get outputs");
  std::vector<HostTensor> outputs;
  outputs.reserve(output_values.size());
  for (size_t index = 0; index < output_values.size(); ++index) {
    ET_CHECK_MSG(output_values[index].isTensor(), "output %zu is not a tensor", index);
    exec_aten::Tensor tensor = output_values[index].toTensor();
    HostTensor host{tensor.scalar_type(), {}, std::vector<uint8_t>(tensor.nbytes())};
    host.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
    ET_CHECK_MSG(
        cudaMemcpy(host.data.data(), tensor.const_data_ptr(), host.data.size(), cudaMemcpyDefault) == cudaSuccess,
        "failed to copy output %zu",
        index);
    outputs.push_back(std::move(host));
  }
  return outputs;
}

} // namespace

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  const char* prefill_path = get_flag(argc, argv, "--prefill_model_path", "pi05_language_prefill_edge.pte");
  const char* decode_path = get_flag(argc, argv, "--decode_model_path", "pi05_language_decode_edge.pte");

  cudaStream_t stream = nullptr;
  ET_CHECK_MSG(cudaStreamCreate(&stream) == cudaSuccess, "failed to create caller stream");
  std::vector<HostTensor> prefill_outputs = run_program(prefill_path, nullptr, stream);
  std::vector<HostTensor> decode_outputs = run_program(decode_path, &prefill_outputs, stream);
  ET_CHECK_MSG(cudaStreamDestroy(stream) == cudaSuccess, "failed to destroy caller stream");

  ET_CHECK_MSG(decode_outputs.size() == 4, "decode must return four outputs");
  ET_CHECK_MSG(
      decode_outputs[1].sizes.size() == 3 && decode_outputs[1].sizes[1] == 1,
      "decode hidden state must contain exactly one token");
  std::fprintf(
      stderr,
      "[edge-llm-e2e] PASS: prefill produced %zu outputs and one-token decode produced %zu outputs\n",
      prefill_outputs.size(),
      decode_outputs.size());
  return 0;
}
