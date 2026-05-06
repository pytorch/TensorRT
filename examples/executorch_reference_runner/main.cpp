/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * C++ inference runner for .pte files compiled with Torch-TensorRT.
 *
 * Usage:
 *   my_runner --model_path=model.pte [--num_runs=1]
 *
 * The runner fills all inputs with ones, runs inference, and prints output
 * shape and sample values.
 */

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

using executorch::extension::FileDataLoader;
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

static uint8_t method_allocator_pool[4 * 1024U * 1024U];
static uint8_t temp_allocator_pool[1 * 1024U * 1024U];

static const char* get_flag(int argc, char** argv, const char* flag, const char* def) {
  const size_t n = strlen(flag);
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], flag, n) == 0 && argv[i][n] == '=') {
      return argv[i] + n + 1;
    }
  }
  return def;
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  const char* model_path = get_flag(argc, argv, "--model_path", "model.pte");
  const int num_runs = atoi(get_flag(argc, argv, "--num_runs", "1"));

  Result<FileDataLoader> loader_result = FileDataLoader::from(model_path);
  if (!loader_result.ok()) {
    ET_LOG(
        Error,
        "FileDataLoader::from('%s') failed: 0x%" PRIx32,
        model_path,
        static_cast<uint32_t>(loader_result.error()));
    return 1;
  }
  auto loader = std::make_unique<FileDataLoader>(std::move(loader_result.get()));

  Result<Program> program = Program::load(loader.get());
  if (!program.ok()) {
    ET_LOG(Error, "Failed to parse model '%s'", model_path);
    return 1;
  }
  ET_LOG(Info, "Model '%s' loaded.", model_path);

  auto name_result = program->get_method_name(0);
  ET_CHECK_MSG(name_result.ok(), "Program has no methods");
  const char* method_name = *name_result;
  ET_LOG(Info, "Method: '%s'", method_name);

  Result<MethodMeta> method_meta = program->method_meta(method_name);
  ET_CHECK_MSG(
      method_meta.ok(),
      "method_meta('%s') failed: 0x%" PRIx32,
      method_name,
      static_cast<uint32_t>(method_meta.error()));

  MemoryAllocator method_allocator{MemoryAllocator(sizeof(method_allocator_pool), method_allocator_pool)};
  MemoryAllocator temp_allocator{MemoryAllocator(sizeof(temp_allocator_pool), temp_allocator_pool)};

  std::vector<std::unique_ptr<uint8_t[]>> planned_buffers;
  std::vector<Span<uint8_t>> planned_spans;
  const size_t num_planned = method_meta->num_memory_planned_buffers();
  for (size_t i = 0; i < num_planned; ++i) {
    const size_t sz = static_cast<size_t>(method_meta->memory_planned_buffer_size(i).get());
    ET_LOG(Info, "  planned buffer[%zu] = %zu bytes", i, sz);
    planned_buffers.push_back(std::make_unique<uint8_t[]>(sz));
    planned_spans.push_back({planned_buffers.back().get(), sz});
  }

  HierarchicalAllocator planned_memory{{planned_spans.data(), planned_spans.size()}};
  MemoryManager memory_manager{&method_allocator, &planned_memory, &temp_allocator};

  Result<Method> method = program->load_method(method_name, &memory_manager, nullptr);
  ET_CHECK_MSG(method.ok(), "load_method('%s') failed: 0x%" PRIx32, method_name, static_cast<uint32_t>(method.error()));
  ET_LOG(Info, "Method loaded. inputs=%zu outputs=%zu", method->inputs_size(), method->outputs_size());

  const size_t num_inputs = method_meta->num_inputs();
  std::vector<std::vector<float>> input_data(num_inputs);
  std::vector<std::vector<exec_aten::SizesType>> input_sizes(num_inputs);
  std::vector<std::vector<exec_aten::DimOrderType>> input_dim_order(num_inputs);
  std::vector<std::vector<exec_aten::StridesType>> input_strides(num_inputs);
  std::vector<exec_aten::TensorImpl> input_impls;
  input_impls.reserve(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    Result<TensorInfo> tensor_info = method_meta->input_tensor_meta(i);
    ET_CHECK_MSG(
        tensor_info.ok(), "input_tensor_meta(%zu) failed: 0x%" PRIx32, i, static_cast<uint32_t>(tensor_info.error()));

    const auto& sizes_ref = tensor_info->sizes();
    const ssize_t ndim = static_cast<ssize_t>(sizes_ref.size());

    input_sizes[i].assign(sizes_ref.begin(), sizes_ref.end());
    input_dim_order[i].resize(ndim);
    input_strides[i].resize(ndim);
    for (ssize_t d = 0; d < ndim; ++d) {
      input_dim_order[i][d] = static_cast<exec_aten::DimOrderType>(d);
    }
    exec_aten::StridesType stride = 1;
    for (ssize_t d = ndim - 1; d >= 0; --d) {
      input_strides[i][d] = stride;
      stride *= static_cast<exec_aten::StridesType>(input_sizes[i][d]);
    }

    const size_t numel = static_cast<size_t>(tensor_info->nbytes() / sizeof(float));
    input_data[i].assign(numel, 1.0f);

    fprintf(stderr, "  input[%zu] shape=[", i);
    for (ssize_t d = 0; d < ndim; ++d) {
      fprintf(stderr, "%d%s", input_sizes[i][d], d + 1 < ndim ? "," : "");
    }
    fprintf(stderr, "] numel=%zu\n", numel);

    input_impls.emplace_back(
        tensor_info->scalar_type(),
        ndim,
        input_sizes[i].data(),
        input_data[i].data(),
        input_dim_order[i].data(),
        input_strides[i].data());
  }

  for (int run = 0; run < num_runs; ++run) {
    for (size_t i = 0; i < num_inputs; ++i) {
      exec_aten::Tensor input_tensor(&input_impls[i]);
      EValue input_evalue(input_tensor);
      Error err = method->set_input(input_evalue, i);
      ET_CHECK_MSG(err == Error::Ok, "set_input(%zu) failed: 0x%" PRIx32, i, static_cast<uint32_t>(err));
    }

    Error status = method->execute();
    ET_CHECK_MSG(status == Error::Ok, "execute() failed on run %d: 0x%" PRIx32, run, static_cast<uint32_t>(status));
  }
  ET_LOG(Info, "Inference completed (%d run(s)).", num_runs);

  const size_t num_outputs = method->outputs_size();
  std::vector<EValue> outputs(num_outputs);
  Error status = method->get_outputs(outputs.data(), num_outputs);
  ET_CHECK_MSG(status == Error::Ok, "get_outputs() failed");

  for (size_t i = 0; i < num_outputs; ++i) {
    if (!outputs[i].isTensor()) {
      ET_LOG(Info, "output[%zu]: not a tensor", i);
      continue;
    }
    exec_aten::Tensor t = outputs[i].toTensor();
    fprintf(stderr, "output[%zu] shape=[", i);
    for (ssize_t d = 0; d < t.dim(); ++d) {
      fprintf(stderr, "%d%s", static_cast<int>(t.size(d)), d + 1 < t.dim() ? "," : "");
    }
    fprintf(stderr, "] numel=%zu dtype=%d\n", static_cast<size_t>(t.numel()), static_cast<int>(t.scalar_type()));

    if (t.scalar_type() == exec_aten::ScalarType::Float) {
      const float* data = t.const_data_ptr<float>();
      const size_t print_n = t.numel() < 8 ? static_cast<size_t>(t.numel()) : 8;
      fprintf(stderr, "  first %zu values:", print_n);
      for (size_t j = 0; j < print_n; ++j) {
        fprintf(stderr, " %.4f", data[j]);
      }
      fprintf(stderr, "\n");
    }
  }

  return 0;
}
