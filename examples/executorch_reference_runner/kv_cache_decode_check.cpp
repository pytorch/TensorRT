/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Caller-owned KV-cache persistence check for a Torch-TensorRT ExecuTorch .pte.
 *
 * Exercises the caller-owned KV contract: the engine's aliased KV output is
 * bound in place to the caller's mutable buffer, which persists across
 * execute() calls. Given a single-layer decode .pte (see
 * examples/torchtrt_executorch_example/export_kv_cache_decode.py) with signature
 * forward(tokens[1,1], input_pos[1]) -> logits, this runs two scenarios on
 * FRESH method loads (each starts from a zeroed cache):
 *
 *   A) one decode at input_pos=1                  (no prior write at pos 0)
 *   B) a decode at input_pos=0, then at input_pos=1
 *
 * At input_pos=1 the causal attention covers positions 0..1. If the cache is
 * shared across execute() calls, scenario B's second step sees the key/value
 * step 0 wrote at position 0, so its logits differ from scenario A (whose
 * position-0 slot is still zero). Equal logits mean the update did not persist
 * (cache reset per call, or the aliased output bound to scratch), so we fail.
 *
 * Usage:
 *   kv_cache_decode_check --model_path=kv_cache_decode.pte [--tol=1e-3]
 */

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

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

static const char* get_flag(int argc, char** argv, const char* flag, const char* def) {
  const size_t n = strlen(flag);
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], flag, n) == 0 && argv[i][n] == '=') {
      return argv[i] + n + 1;
    }
  }
  return def;
}

// Load a FRESH method (zeroed caller-owned buffers), run one decode step per
// entry in `positions` (token id fixed to 1, input_pos = the position), and
// return the final step's first output as host floats.
static std::vector<float> run_decode(Program& program, const char* method_name, const std::vector<int64_t>& positions) {
  Result<MethodMeta> method_meta = program.method_meta(method_name);
  ET_CHECK_MSG(method_meta.ok(), "method_meta failed: 0x%" PRIx32, static_cast<uint32_t>(method_meta.error()));

  auto method_pool = std::make_unique<uint8_t[]>(4 * 1024U * 1024U);
  auto temp_pool = std::make_unique<uint8_t[]>(1 * 1024U * 1024U);
  MemoryAllocator method_allocator{4 * 1024U * 1024U, method_pool.get()};
  MemoryAllocator temp_allocator{1 * 1024U * 1024U, temp_pool.get()};

  // Caller-owned KV buffers live in the memory-planned arenas. Arenas tagged
  // CUDA (by PropagateDevicePass, for device tensors a delegate reads/writes)
  // must be backed by real device memory -- otherwise the aliased KV input is
  // not device-resident and the backend rejects it.
  std::vector<std::unique_ptr<uint8_t[]>> host_arenas;
  std::vector<void*> cuda_arenas;
  std::vector<Span<uint8_t>> planned_spans;
  const size_t num_planned = method_meta->num_memory_planned_buffers();
  for (size_t i = 0; i < num_planned; ++i) {
    const size_t sz = static_cast<size_t>(method_meta->memory_planned_buffer_size(i).get());
    auto dev = method_meta->memory_planned_buffer_device(i);
    if (dev.ok() && dev.get().type() == executorch::runtime::etensor::DeviceType::CUDA) {
      void* p = nullptr;
      ET_CHECK_MSG(cudaMalloc(&p, sz) == cudaSuccess, "cudaMalloc planned buffer %zu failed", i);
      cuda_arenas.push_back(p);
      planned_spans.push_back({reinterpret_cast<uint8_t*>(p), sz});
    } else {
      host_arenas.push_back(std::make_unique<uint8_t[]>(sz));
      planned_spans.push_back({host_arenas.back().get(), sz});
    }
  }
  HierarchicalAllocator planned_memory{{planned_spans.data(), planned_spans.size()}};
  MemoryManager memory_manager{&method_allocator, &planned_memory, &temp_allocator};

  Result<Method> method = program.load_method(method_name, &memory_manager, nullptr);
  ET_CHECK_MSG(method.ok(), "load_method failed: 0x%" PRIx32, static_cast<uint32_t>(method.error()));

  // One int64 tensor per declared input (numel==1 for a decode step): input 0 is
  // the token id, the rest carry input_pos. Rank is read from method_meta so this
  // works whether input_pos is rank-1 ([1]) or rank-2 ([1,1]).
  const size_t num_inputs = method_meta->num_inputs();
  std::vector<std::vector<int64_t>> data(num_inputs);
  std::vector<std::vector<exec_aten::SizesType>> sizes(num_inputs);
  std::vector<std::vector<exec_aten::DimOrderType>> dim_order(num_inputs);
  std::vector<std::vector<exec_aten::StridesType>> strides(num_inputs);
  std::vector<exec_aten::TensorImpl> impls;
  impls.reserve(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    Result<TensorInfo> ti = method_meta->input_tensor_meta(i);
    ET_CHECK_MSG(ti.ok(), "input_tensor_meta(%zu) failed", i);
    const auto& s = ti->sizes();
    const ssize_t nd = static_cast<ssize_t>(s.size());
    sizes[i].assign(s.begin(), s.end());
    dim_order[i].resize(nd);
    strides[i].resize(nd);
    exec_aten::StridesType stride = 1;
    for (ssize_t d = nd - 1; d >= 0; --d) {
      dim_order[i][d] = static_cast<exec_aten::DimOrderType>(d);
      strides[i][d] = stride;
      stride *= static_cast<exec_aten::StridesType>(sizes[i][d]);
    }
    size_t numel = 1;
    for (auto x : sizes[i])
      numel *= static_cast<size_t>(x);
    data[i].assign(numel, i == 0 ? 1 : 0);
    impls.emplace_back(
        exec_aten::ScalarType::Long, nd, sizes[i].data(), data[i].data(), dim_order[i].data(), strides[i].data());
  }

  for (int64_t pos : positions) {
    for (size_t i = 1; i < num_inputs; ++i) {
      std::fill(data[i].begin(), data[i].end(), pos);
    }
    for (size_t i = 0; i < num_inputs; ++i) {
      ET_CHECK(method->set_input(EValue(exec_aten::Tensor(&impls[i])), i) == Error::Ok);
    }
    ET_CHECK_MSG(method->execute() == Error::Ok, "execute() failed at pos %" PRId64, pos);
  }

  EValue out;
  ET_CHECK_MSG(method->get_outputs(&out, 1) == Error::Ok, "get_outputs failed");
  ET_CHECK_MSG(out.isTensor(), "output 0 is not a tensor");
  exec_aten::Tensor t = out.toTensor();
  ET_CHECK_MSG(t.scalar_type() == exec_aten::ScalarType::Float, "expected float logits output");
  // The output may be device-resident; cudaMemcpyDefault copies from host or
  // device. execute() synchronized (no caller stream) so the result is ready.
  std::vector<float> result(static_cast<size_t>(t.numel()));
  ET_CHECK_MSG(
      cudaMemcpy(result.data(), t.const_data_ptr(), result.size() * sizeof(float), cudaMemcpyDefault) == cudaSuccess,
      "cudaMemcpy of logits to host failed");
  for (void* p : cuda_arenas) {
    cudaFree(p);
  }
  return result;
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  const char* model_path = get_flag(argc, argv, "--model_path", "kv_cache_decode.pte");
  const double tol = atof(get_flag(argc, argv, "--tol", "1e-3"));

  Result<FileDataLoader> loader = FileDataLoader::from(model_path);
  ET_CHECK_MSG(loader.ok(), "FileDataLoader::from('%s') failed", model_path);
  auto loader_ptr = std::make_unique<FileDataLoader>(std::move(loader.get()));
  Result<Program> program = Program::load(loader_ptr.get());
  ET_CHECK_MSG(program.ok(), "Failed to parse model '%s'", model_path);

  auto name = program->get_method_name(0);
  ET_CHECK_MSG(name.ok(), "Program has no methods");
  const char* method_name = *name;
  ET_LOG(Info, "Loaded '%s' method '%s'", model_path, method_name);

  // A: pos=1 from a zeroed cache. B: pos=0 then pos=1 (second step sees pos 0).
  std::vector<float> a = run_decode(*program, method_name, {1});
  std::vector<float> b = run_decode(*program, method_name, {0, 1});

  ET_CHECK_MSG(a.size() == b.size() && !a.empty(), "output size mismatch (%zu vs %zu)", a.size(), b.size());
  double max_abs_diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }

  fprintf(
      stderr,
      "[kv-check] logits numel=%zu  max|A(no-history) - B(with-history)| = %.6g  (tol=%.3g)\n",
      a.size(),
      max_abs_diff,
      tol);
  if (max_abs_diff > tol) {
    fprintf(stderr, "[kv-check] PASS: decode at pos=1 observed the KV written at pos=0 across execute() calls.\n");
    return 0;
  }
  fprintf(stderr, "[kv-check] FAIL: outputs are identical -> the KV write did not persist across execute() calls.\n");
  return 1;
}
