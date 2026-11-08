/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "execution_graph_cuda_wrappers.h"
#include "torch_tensorrt/executorch/EngineHandle.h"
#include "torch_tensorrt/executorch/ExecutionGraph.h"
#include "torch_tensorrt/executorch/SharedScratchPool.h"
#include "torch_tensorrt/executorch/SharedScratchPoolTestHooks.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <cudaTypedefs.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <thread>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using namespace ::executorch::aten;
using namespace ::executorch::runtime;
using namespace ::executorch::extension::cuda;
using namespace torch_tensorrt::executorch_backend::testing;

// Keep these keys literal so a backend rename makes the test fail.
constexpr char kGraphOption[] = "use_cuda_graphs";
constexpr char kScratchOption[] = "use_shared_activation_scratch";
constexpr size_t kMaxElements = 8 * 32;
constexpr size_t kMaxBytes = kMaxElements * sizeof(float);

BackendOption option(const char* key, bool value) {
  BackendOption result{};
  std::strncpy(result.key, key, sizeof(result.key) - 1);
  result.value = value;
  return result;
}

Error set_options(Span<BackendOption> options) {
  TensorRTBackend backend;
  BackendOptionContext context;
  return backend.set_option(context, options);
}

Error set_graphs(bool enabled) {
  BackendOption options[] = {option(kGraphOption, enabled)};
  return set_options({options, 1});
}

std::vector<uint8_t> wrap_plan(const nvinfer1::IHostMemory& plan, const std::string& metadata, bool alias) {
  const uint32_t metadata_offset = 32;
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const uint32_t engine_offset = (metadata_offset + metadata_size + 15) / 16 * 16;
  const uint64_t engine_size = plan.size();
  std::vector<uint8_t> blob(engine_offset + engine_size, 0);
  std::memcpy(blob.data(), alias ? "TR02" : "TR01", 4);
  std::memcpy(blob.data() + 4, &metadata_offset, sizeof(metadata_offset));
  std::memcpy(blob.data() + 8, &metadata_size, sizeof(metadata_size));
  std::memcpy(blob.data() + 12, &engine_offset, sizeof(engine_offset));
  std::memcpy(blob.data() + 16, &engine_size, sizeof(engine_size));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(blob.data() + engine_offset, plan.data(), plan.size());
  return blob;
}

std::vector<uint8_t> build_blob(bool alias, int profiles = 1) {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, -1});
  const float scale = 2.0f;
  const float bias = 1.0f;
  auto* multiplier = network->addConstant({2, {1, 1}}, {nvinfer1::DataType::kFLOAT, &scale, 1});
  auto* addend = network->addConstant({2, {1, 1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
  if (!input || !multiplier || !addend) {
    return {};
  }
  auto* product = network->addElementWise(*input, *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  if (!product) {
    return {};
  }
  auto* sum =
      network->addElementWise(*product->getOutput(0), *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
  if (!sum) {
    return {};
  }
  sum->getOutput(0)->setName("output_0");
  network->markOutput(*sum->getOutput(0));
  for (int p = 0; p < profiles; ++p) {
    auto* profile = builder->createOptimizationProfile();
    if (!profile || !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 2}) ||
        !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{2, 8}) ||
        !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{8, 32}) ||
        config->addOptimizationProfile(profile) < 0) {
      return {};
    }
  }
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0)";
  if (alias) {
    metadata += R"(,"aliased_io":[{"output":"output_0","input":"input_0","kind":"user"}])";
  }
  metadata += "}";
  return wrap_plan(*plan, metadata, alias);
}

// Two softmaxes over different axes keep the chain from fusing into one pass, so the engine needs scratch.
std::vector<uint8_t> build_scratch_blob(int32_t rows, int32_t cols, bool alias = false) {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{rows, cols});
  auto* over_cols = input ? network->addSoftMax(*input) : nullptr;
  if (!over_cols) {
    return {};
  }
  over_cols->setAxes(1u << 1);
  auto* over_rows = network->addSoftMax(*over_cols->getOutput(0));
  if (!over_rows) {
    return {};
  }
  over_rows->setAxes(1u << 0);
  over_rows->getOutput(0)->setName("output_0");
  network->markOutput(*over_rows->getOutput(0));
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0)";
  if (alias) {
    metadata += R"(,"aliased_io":[{"output":"output_0","input":"input_0","kind":"user"}])";
  }
  metadata += "}";
  return wrap_plan(*plan, metadata, alias);
}

// Two independent branches of two layers each, big enough that TensorRT runs one of them on an auxiliary
// stream it creates with the execution context.
std::vector<uint8_t> build_aux_stream_blob(int32_t rows, int32_t cols) {
  static TRTLogger logger;
  static std::vector<float> weights[4];
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !config || !config->setMaxAuxStreams(1)) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{rows, cols});
  if (!input) {
    return {};
  }
  nvinfer1::ITensor* branches[2] = {};
  for (int b = 0; b < 2; ++b) {
    nvinfer1::ITensor* t = input;
    for (int l = 0; l < 2; ++l) {
      std::vector<float>& w = weights[b * 2 + l];
      w.resize(static_cast<size_t>(cols) * cols);
      for (size_t i = 0; i < w.size(); ++i) {
        w[i] = static_cast<float>(static_cast<int>((i * 31 + b * 7 + l * 3) % 17) - 8) / (8.0f * cols);
      }
      auto* constant = network->addConstant(
          nvinfer1::Dims2{cols, cols}, {nvinfer1::DataType::kFLOAT, w.data(), static_cast<int64_t>(w.size())});
      auto* product = constant
          ? network->addMatrixMultiply(
                *t, nvinfer1::MatrixOperation::kNONE, *constant->getOutput(0), nvinfer1::MatrixOperation::kNONE)
          : nullptr;
      auto* activation =
          product ? network->addActivation(*product->getOutput(0), nvinfer1::ActivationType::kTANH) : nullptr;
      if (!activation) {
        return {};
      }
      t = activation->getOutput(0);
    }
    branches[b] = t;
  }
  auto* sum = network->addElementWise(*branches[0], *branches[1], nvinfer1::ElementWiseOperation::kSUM);
  if (!sum) {
    return {};
  }
  sum->getOutput(0)->setName("output_0");
  network->markOutput(*sum->getOutput(0));
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  return wrap_plan(
      *plan,
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0})",
      false);
}

std::vector<uint8_t> build_kv_blob() {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !config) {
    return {};
  }
  auto* cache = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 4, 4});
  const int32_t index = 1;
  const float bias = 1;
  auto* update = network->addInput("input_1", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 1, 4});
  auto* indices = network->addConstant({1, {1}}, {nvinfer1::DataType::kINT32, &index, 1});
  auto* addend = network->addConstant({4, {1, 1, 1, 1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
  if (!cache || !update || !indices || !addend) {
    return {};
  }
  auto* write = network->addKVCacheUpdate(*cache, *update, *indices->getOutput(0), nvinfer1::KVCacheMode::kLINEAR);
  if (!write) {
    return {};
  }
  write->getOutput(0)->setName("output_0");
  network->markOutput(*write->getOutput(0));
  auto* sum = network->addElementWise(*update, *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
  if (!sum) {
    return {};
  }
  sum->getOutput(0)->setName("output_1");
  network->markOutput(*sum->getOutput(0));
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  return wrap_plan(
      *plan,
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"input_1","is_input":true},{"name":"output_0","is_input":false},{"name":"output_1","is_input":false}],"device_id":0,"aliased_io":[{"output":"output_0","input":"input_0","kind":"kv_cache_update"}]})",
      true);
}

template <typename Function>
Function driver_entry(const char* name, unsigned int version) {
  void* entry = nullptr;
  EXPECT_EQ(cudaGetDriverEntryPointByVersion(name, &entry, version, cudaEnableDefault), cudaSuccess) << name;
  return reinterpret_cast<Function>(entry);
}

class LoadedGraphEngine {
 public:
  ~LoadedGraphEngine() {
    backend_.destroy(handle_);
  }

  Error load(
      const std::vector<uint8_t>& blob,
      ::executorch::runtime::ArrayRef<CompileSpec> specs = {},
      Span<const BackendOption> options = {}) {
    BackendInitContext context(&allocator_, nullptr, nullptr, nullptr, options);
    FreeableBuffer processed(blob.data(), blob.size(), nullptr);
    auto result = backend_.init(context, &processed, specs);
    if (!result.ok()) {
      return result.error();
    }
    handle_ = static_cast<EngineHandle*>(result.get());
    return Error::Ok;
  }

  Error run(
      void* input,
      void* output,
      SizesType rows,
      SizesType cols,
      std::optional<cudaStream_t> stream = std::nullopt) {
    SizesType input_sizes[] = {rows, cols};
    SizesType output_sizes[] = {rows, cols};
    TensorImpl input_impl(ScalarType::Float, 2, input_sizes, input);
    TensorImpl output_impl(ScalarType::Float, 2, output_sizes, output);
    EValue input_value{Tensor(&input_impl)};
    EValue output_value{Tensor(&output_impl)};
    EValue* args[] = {&input_value, &output_value};
    return run({args, 2}, stream);
  }

  Error run(Span<EValue*> args, std::optional<cudaStream_t> stream = std::nullopt) {
    BackendExecutionContext context;
    if (stream.has_value()) {
      CallerStreamGuard guard(*stream);
      return backend_.execute(context, handle_, args);
    }
    return backend_.execute(context, handle_, args);
  }

  EngineHandle* handle() const {
    return handle_;
  }

  bool is_captured() const {
    return handle_->execution_graph != nullptr && handle_->execution_graph->is_captured();
  }

 private:
  TensorRTBackend backend_;
  alignas(EngineHandle) uint8_t arena_[4096]{};
  MemoryAllocator allocator_{sizeof(arena_), arena_};
  EngineHandle* handle_ = nullptr;
};

void concurrent_missing_driver_log() {
  runtime_init();
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  const auto blob = build_blob(false);
  ASSERT_FALSE(blob.empty());
  const BackendOption graphs = option(kGraphOption, true);
  LoadedGraphEngine first, second;
  ASSERT_EQ(first.load(blob, {}, {&graphs, 1}), Error::Ok);
  ASSERT_EQ(second.load(blob, {}, {&graphs, 1}), Error::Ok);
  ASSERT_NE(first.handle()->execution_graph, nullptr);
  ASSERT_NE(second.handle()->execution_graph, nullptr);

  static std::mutex log_gate;
  static std::atomic<int> log_count{0};
  static std::promise<void> logger_entered;
  ASSERT_TRUE(register_pal(PalImpl::create(
      [](et_timestamp_t, et_pal_log_level_t, const char*, const char*, size_t, const char* message, size_t length) {
        if (std::string_view(message, length).find("CUDA graph replay needs a CUDA 12.5 or newer driver") !=
            std::string_view::npos) {
          if (log_count.fetch_add(1) == 0) {
            logger_entered.set_value();
          }
          const std::lock_guard<std::mutex> wait(log_gate);
        }
      },
      __FILE__)));
  std::vector<float> input(16, 3.0f), first_output(16, -1.0f), second_output(16, -1.0f);
  std::promise<Error> completed;
  auto second_result = completed.get_future();
  Error first_result = Error::Internal;
  size_t first_queries = 0, second_queries = 0;
  std::unique_lock<std::mutex> gate(log_gate);
  std::thread stalled([&] {
    CudaCalls calls;
    calls.disable_stream_context = true;
    first_result = first.run(input.data(), first_output.data(), 2, 8);
    first_queries = calls.stream_context_queries;
  });
  constexpr std::chrono::seconds deadline{30};
  const bool reached_log = logger_entered.get_future().wait_for(deadline) == std::future_status::ready;
  std::thread caller([&] {
    CudaCalls calls;
    calls.disable_stream_context = true;
    completed.set_value(second.run(input.data(), second_output.data(), 2, 8));
    second_queries = calls.stream_context_queries;
  });
  const bool finished = reached_log && second_result.wait_for(deadline) == std::future_status::ready;
  // Release before joining, so the old initialization guard fails without hanging the test.
  gate.unlock();
  stalled.join();
  caller.join();
  ASSERT_TRUE(reached_log);
  EXPECT_TRUE(finished) << "an independent caller waited for the blocked old-driver log";
  EXPECT_EQ(first_result, Error::Ok);
  EXPECT_EQ(second_result.get(), Error::Ok);
  EXPECT_EQ(first_queries, 1u);
  EXPECT_EQ(second_queries, 0u);
  EXPECT_EQ(first_output, std::vector<float>(16, 7.0f));
  EXPECT_EQ(second_output, first_output);
  EXPECT_EQ(second.run(input.data(), second_output.data(), 2, 8), Error::Ok);
  EXPECT_EQ(log_count.load(), 1);
  EXPECT_FALSE(first.is_captured());
  EXPECT_FALSE(second.is_captured());
}

TEST(ExecutionGraphDeathTest, ConcurrentCallerFinishesWhileMissingDriverLogIsBlocked) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    const char* required = std::getenv("TORCHTRT_EXECUTORCH_REQUIRE_CUDA");
    if (required && std::strcmp(required, "1") == 0) {
      FAIL() << "This run requires CUDA; the old-driver concurrency test could not execute";
    }
    GTEST_SKIP() << "No CUDA device; old-driver concurrency is not covered";
  }
  int pools = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0), cudaSuccess);
  if (!pools) {
    GTEST_SKIP() << "This GPU has no stream-ordered memory, so replay stays off";
  }
  // Re-exec keeps the cached entry and once-only log fresh, including under --gtest_repeat.
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  ASSERT_EXIT(
      {
        concurrent_missing_driver_log();
        std::exit(::testing::Test::HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

class ExecutionGraphTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      return;
    }
    if (!blob_.empty()) {
      return;
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    int pools = 0;
    memory_pools_ = cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, 0) == cudaSuccess && pools != 0;
    blob_ = build_blob(false);
    alias_blob_ = build_blob(true);
    kv_blob_ = build_kv_blob();
    ASSERT_FALSE(blob_.empty());
    ASSERT_FALSE(alias_blob_.empty());
    // Observe the default before any test changes the process-wide option.
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob_), Error::Ok);
    default_has_graph_ = engine.handle()->execution_graph != nullptr;
    char on[] = "1";
    CompileSpec spec{kGraphOption, {on, 1}};
    LoadedGraphEngine saved;
    ASSERT_EQ(saved.load(blob_, {&spec, 1}), Error::Ok);
    saved_default_has_graph_ = saved.handle()->execution_graph != nullptr;
  }

  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      const char* required = std::getenv("TORCHTRT_EXECUTORCH_REQUIRE_CUDA");
      if (required && std::strcmp(required, "1") == 0) {
        FAIL() << "This run requires CUDA; no graph test could execute";
      }
      GTEST_SKIP() << "No CUDA device; graph execution is not covered";
    }
    ASSERT_FALSE(blob_.empty());
    ASSERT_FALSE(alias_blob_.empty());
    ASSERT_EQ(cudaGetDevice(&original_device_), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    BackendOption options[] = {option(kScratchOption, false), option(kGraphOption, true)};
    ASSERT_EQ(set_options({options, 2}), Error::Ok);
    for (int i = 0; i < 2; ++i) {
      ASSERT_EQ(cudaMalloc(&inputs_[i], kMaxBytes), cudaSuccess);
      ASSERT_EQ(cudaMalloc(&outputs_[i], kMaxBytes), cudaSuccess);
      ASSERT_EQ(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking), cudaSuccess);
    }
    ASSERT_NE(inputs_[0], inputs_[1]);
    ASSERT_NE(outputs_[0], outputs_[1]);
    ASSERT_EQ(cudaMalloc(&reference_output_, kMaxBytes), cudaSuccess);
  }

  void TearDown() override {
    if (original_device_ < 0) {
      return;
    }
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
    for (int i = 0; i < 2; ++i) {
      if (streams_[i]) {
        EXPECT_EQ(cudaStreamDestroy(streams_[i]), cudaSuccess);
      }
      EXPECT_EQ(cudaFree(inputs_[i]), cudaSuccess);
      EXPECT_EQ(cudaFree(outputs_[i]), cudaSuccess);
    }
    EXPECT_EQ(cudaFree(reference_output_), cudaSuccess);
    BackendOption options[] = {option(kScratchOption, false), option(kGraphOption, false)};
    EXPECT_EQ(set_options({options, 2}), Error::Ok);
    EXPECT_EQ(cudaSetDevice(original_device_), cudaSuccess);
  }

  void load_graph(LoadedGraphEngine& graph) {
    ASSERT_EQ(graph.load(blob_), Error::Ok);
    ASSERT_NE(graph.handle()->execution_graph, nullptr);
  }

  void load_pair(LoadedGraphEngine& plain, LoadedGraphEngine& graph) {
    ASSERT_EQ(set_graphs(false), Error::Ok);
    ASSERT_EQ(plain.load(blob_), Error::Ok);
    ASSERT_EQ(plain.handle()->execution_graph, nullptr);
    ASSERT_EQ(set_graphs(true), Error::Ok);
    ASSERT_NO_FATAL_FAILURE(load_graph(graph));
    ASSERT_NE(graph.handle()->execution_graph, nullptr);
    ASSERT_FALSE(graph.is_captured());
  }

  void compare(
      LoadedGraphEngine& plain,
      LoadedGraphEngine& graph,
      int slot,
      SizesType rows,
      SizesType cols,
      int seed,
      cudaStream_t stream) {
    const size_t count = static_cast<size_t>(rows) * cols;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
      input[i] = static_cast<float>(static_cast<int>(i % 17) - 8 + seed * 3) / 4.0f;
    }
    ASSERT_EQ(
        cudaMemcpyAsync(inputs_[slot], input.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream),
        cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(outputs_[slot], 0xff, kMaxBytes, stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(reference_output_, 0xff, kMaxBytes, stream), cudaSuccess);
    ASSERT_EQ(plain.run(inputs_[slot], reference_output_, rows, cols, stream), Error::Ok);
    ASSERT_EQ(graph.run(inputs_[slot], outputs_[slot], rows, cols, stream), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::vector<float> actual(count), expected(count);
    ASSERT_EQ(cudaMemcpy(actual.data(), outputs_[slot], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(expected.data(), reference_output_, count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, expected);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_FLOAT_EQ(actual[i], input[i] * 2.0f + 1.0f) << "element " << i;
    }
    EXPECT_EQ(plain.handle()->execution_graph, nullptr);
  }

  void compare_default(LoadedGraphEngine& graph, int slot, SizesType rows, SizesType cols, int seed) {
    ASSERT_FALSE(getCallerStream().has_value());
    const size_t count = static_cast<size_t>(rows) * cols;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i) {
      input[i] = static_cast<float>(seed * 16 + i);
    }
    ASSERT_EQ(
        cudaMemcpyAsync(
            inputs_[slot], input.data(), count * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread),
        cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(outputs_[slot], 0xff, kMaxBytes, cudaStreamPerThread), cudaSuccess);
    ASSERT_EQ(graph.run(inputs_[slot], outputs_[slot], rows, cols), Error::Ok);
    EXPECT_FALSE(getCallerStream().has_value());
    std::vector<float> output(count);
    ASSERT_EQ(cudaMemcpy(output.data(), outputs_[slot], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_FLOAT_EQ(output[i], input[i] * 2.0f + 1.0f) << "element " << i;
    }
  }

  // Alternates the two slots until the graph gives up following the caller's addresses: a warm-up,
  // a recording on slot 0, a recording on slot 1 when they move, a recording on the graph's own buffers
  // when they move again, and a replay on those buffers.
  void stage(LoadedGraphEngine& plain, LoadedGraphEngine& graph, SizesType rows, SizesType cols, cudaStream_t stream) {
    const int slots[] = {0, 0, 1, 0, 1};
    for (int i = 0; i < 5; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, slots[i], rows, cols, i, stream));
    }
    ASSERT_TRUE(graph.is_captured());
    ASSERT_NE(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[1]);
  }

  void stage_default(LoadedGraphEngine& graph, SizesType rows, SizesType cols) {
    const int slots[] = {0, 0, 1, 0, 1};
    for (int i = 0; i < 5; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare_default(graph, slots[i], rows, cols, i));
    }
    ASSERT_TRUE(graph.is_captured());
    ASSERT_NE(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[1]);
  }

  static std::vector<uint8_t> blob_;
  static std::vector<uint8_t> alias_blob_;
  static std::vector<uint8_t> kv_blob_;
  static bool default_has_graph_;
  static bool saved_default_has_graph_;
  static bool memory_pools_;
  int original_device_ = -1;
  void* inputs_[2]{};
  void* outputs_[2]{};
  void* reference_output_ = nullptr;
  cudaStream_t streams_[2]{};
};

std::vector<uint8_t> ExecutionGraphTest::blob_;
std::vector<uint8_t> ExecutionGraphTest::alias_blob_;
std::vector<uint8_t> ExecutionGraphTest::kv_blob_;
bool ExecutionGraphTest::default_has_graph_ = false;
bool ExecutionGraphTest::saved_default_has_graph_ = false;
bool ExecutionGraphTest::memory_pools_ = false;

// Replay needs stream-ordered memory. Without it the backend keeps enqueueV3, which the
// ExecutionGraphTest cases cover.
class ExecutionGraphReplayTest : public ExecutionGraphTest {
 protected:
  void SetUp() override {
    ExecutionGraphTest::SetUp();
    if (!IsSkipped() && !HasFatalFailure() && !memory_pools_) {
      GTEST_SKIP() << "This GPU has no stream-ordered memory, so replay stays off";
    }
  }
};

TEST_F(ExecutionGraphTest, DisabledByDefault) {
  EXPECT_FALSE(default_has_graph_);
  EXPECT_EQ(saved_default_has_graph_, memory_pools_);
}

// output_i = input_i * 2 + 1 for three independent 1-D float inputs of up to 1024 elements, so each output can
// be placed anywhere against every input.
std::vector<uint8_t> build_three_path_blob() {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  auto* profile = builder->createOptimizationProfile();
  if (!network || !config || !profile) {
    return {};
  }
  static const float scale = 2.0f;
  static const float bias = 1.0f;
  std::string bindings;
  for (int path = 0; path < 3; ++path) {
    const std::string suffix = std::to_string(path);
    auto* input = network->addInput(("input_" + suffix).c_str(), nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {-1}});
    auto* multiplier = network->addConstant(nvinfer1::Dims{1, {1}}, {nvinfer1::DataType::kFLOAT, &scale, 1});
    auto* addend = network->addConstant(nvinfer1::Dims{1, {1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
    if (!input || !multiplier || !addend) {
      return {};
    }
    auto* product = network->addElementWise(*input, *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    auto* sum = product
        ? network->addElementWise(*product->getOutput(0), *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)
        : nullptr;
    if (!sum || !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}}) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {256}}) ||
        !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {1024}})) {
      return {};
    }
    sum->getOutput(0)->setName(("output_" + suffix).c_str());
    network->markOutput(*sum->getOutput(0));
    bindings += R"({"name":"input_)" + suffix + R"(","is_input":true},)";
  }
  for (int path = 0; path < 3; ++path) {
    bindings += R"({"name":"output_)" + std::to_string(path) + R"(","is_input":false})" + (path < 2 ? "," : "");
  }
  if (config->addOptimizationProfile(profile) < 0) {
    return {};
  }
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  return wrap_plan(*plan, R"({"io_bindings":[)" + bindings + R"(],"device_id":0})", false);
}

// Runs the three-path engine once, without replay, on views of one device arena given as offsets and lengths in
// floats, then checks that output_0 went through a buffer of the engine exactly when `overlaps`, and that every
// output holds 2 * x + 1 for each x its input held before the call.
void expect_output_0_staged_when_overlapping(
    const std::vector<size_t>& in,
    const std::vector<size_t>& length,
    const std::vector<size_t>& out,
    bool overlaps,
    cudaStream_t stream) {
  static const std::vector<uint8_t> blob = build_three_path_blob();
  ASSERT_FALSE(blob.empty());
  ASSERT_EQ(set_graphs(false), Error::Ok);
  LoadedGraphEngine engine;
  ASSERT_EQ(engine.load(blob), Error::Ok);
  constexpr size_t kArena = 4096;
  void* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, kArena * sizeof(float)), cudaSuccess);
  std::unique_ptr<void, decltype(&cudaFree)> arena_guard(raw, &cudaFree);
  float* const arena = static_cast<float*>(raw);
  std::vector<float> before(kArena);
  for (size_t i = 0; i < kArena; ++i) {
    before[i] = static_cast<float>(i);
  }
  ASSERT_EQ(cudaMemcpyAsync(arena, before.data(), kArena * sizeof(float), cudaMemcpyHostToDevice, stream), cudaSuccess);
  SizesType sizes[] = {
      static_cast<SizesType>(length[0]), static_cast<SizesType>(length[1]), static_cast<SizesType>(length[2])};
  DimOrderType dim_order[] = {0};
  StridesType strides[] = {1};
  TensorImpl input_0(ScalarType::Float, 1, &sizes[0], arena + in[0], dim_order, strides);
  TensorImpl input_1(ScalarType::Float, 1, &sizes[1], arena + in[1], dim_order, strides);
  TensorImpl input_2(ScalarType::Float, 1, &sizes[2], arena + in[2], dim_order, strides);
  TensorImpl output_0(ScalarType::Float, 1, &sizes[0], arena + out[0], dim_order, strides);
  TensorImpl output_1(ScalarType::Float, 1, &sizes[1], arena + out[1], dim_order, strides);
  TensorImpl output_2(ScalarType::Float, 1, &sizes[2], arena + out[2], dim_order, strides);
  EValue values[] = {
      EValue(Tensor(&input_0)),
      EValue(Tensor(&input_1)),
      EValue(Tensor(&input_2)),
      EValue(Tensor(&output_0)),
      EValue(Tensor(&output_1)),
      EValue(Tensor(&output_2))};
  EValue* args[] = {&values[0], &values[1], &values[2], &values[3], &values[4], &values[5]};
  ASSERT_EQ(engine.run({args, 6}, stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(engine.handle()->exec_ctx->getTensorAddress("output_0") != arena + out[0], overlaps);
  std::vector<float> after(kArena);
  ASSERT_EQ(cudaMemcpy(after.data(), arena, kArena * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t p = 0; p < 3; ++p) {
    for (size_t i = 0; i < length[p]; ++i) {
      ASSERT_FLOAT_EQ(after[out[p] + i], before[in[p] + i] * 2.0f + 1.0f) << "output_" << p << " element " << i;
    }
  }
}

// Offsets below are in floats, and 64 floats are 256 bytes, so every view is aligned and bound directly unless it
// overlaps an input. output_1 and output_2 always sit apart from everything.

// An output can overlap a large input past the end of a smaller input that starts inside the large one.
TEST_F(ExecutionGraphReplayTest, AnOutputOverlappingAnInputPastTheEndOfAnotherInputIsStaged) {
  expect_output_0_staged_when_overlapping({0, 64, 2048}, {512, 64, 64}, {256, 3072, 3328}, true, streams_[0]);
}

// Inputs can come in any order of address, so an output that overlaps the last input, which lies below the
// others, is staged too.
TEST_F(ExecutionGraphReplayTest, AnOutputOverlappingAnInputAtALowerAddressThanTheOthersIsStaged) {
  expect_output_0_staged_when_overlapping({2048, 2560, 0}, {256, 256, 256}, {128, 3072, 3584}, true, streams_[0]);
}

// An output that starts inside an input, or ends inside one, overlaps it.
TEST_F(ExecutionGraphReplayTest, AnOutputThatPartlyOverlapsAnInputIsStaged) {
  {
    SCOPED_TRACE("starts inside");
    expect_output_0_staged_when_overlapping({0, 2048, 2304}, {256, 64, 64}, {64, 3072, 3328}, true, streams_[0]);
  }
  {
    SCOPED_TRACE("ends inside");
    expect_output_0_staged_when_overlapping({128, 2048, 2304}, {256, 64, 64}, {0, 3072, 3328}, true, streams_[0]);
  }
}

// Ranges that only touch share no memory, so the output stays bound directly.
TEST_F(ExecutionGraphReplayTest, AnOutputThatOnlyTouchesAnInputIsBoundDirectly) {
  {
    SCOPED_TRACE("output ends where the input starts");
    expect_output_0_staged_when_overlapping({256, 2048, 2304}, {256, 64, 64}, {0, 3072, 3328}, false, streams_[0]);
  }
  {
    SCOPED_TRACE("input ends where the output starts");
    expect_output_0_staged_when_overlapping({0, 2048, 2304}, {256, 64, 64}, {256, 3072, 3328}, false, streams_[0]);
  }
}

// A kernel can write an output before it has read all of an input, so an output that shares memory with an
// input is written to a buffer of the engine and copied over after the engine finishes, with or without
// replay. An output next to an input is still bound directly.
TEST_F(ExecutionGraphReplayTest, AnOutputThatOverlapsAnInputIsStaged) {
  for (bool replay : {false, true}) {
    SCOPED_TRACE(replay ? "replay" : "plain");
    ASSERT_EQ(set_graphs(replay), Error::Ok);
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob_), Error::Ok);
    ASSERT_EQ(engine.handle()->execution_graph != nullptr, replay);
    std::vector<float> input(16), output(16);
    for (int call = 0; call < 4; ++call) {
      SCOPED_TRACE(call);
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i + call * 16);
      }
      ASSERT_EQ(
          cudaMemcpyAsync(inputs_[0], input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice, streams_[0]),
          cudaSuccess);
      ASSERT_EQ(engine.run(inputs_[0], inputs_[0], 2, 8, streams_[0]), Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
      EXPECT_NE(engine.handle()->exec_ctx->getTensorAddress("output_0"), inputs_[0]);
      ASSERT_EQ(
          cudaMemcpy(output.data(), inputs_[0], sizeof(float) * output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
      for (size_t i = 0; i < output.size(); ++i) {
        ASSERT_FLOAT_EQ(output[i], input[i] * 2.0f + 1.0f) << "element " << i;
      }
    }
    EXPECT_EQ(engine.is_captured(), replay);
    // 2x8 floats end 64 bytes in, well before the next 256-byte boundary.
    void* next_to_input = static_cast<char*>(inputs_[0]) + 256;
    ASSERT_EQ(engine.run(inputs_[0], next_to_input, 2, 8, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    EXPECT_EQ(engine.handle()->exec_ctx->getTensorAddress("output_0"), next_to_input);
  }
}

TEST_F(ExecutionGraphReplayTest, LoadOptionWinsOverProcessRefusalAndSavedChoice) {
  char on[] = "1";
  char off[] = "0";
  for (bool process_option : {false, true}) {
    for (char* value : {static_cast<char*>(nullptr), off, on}) {
      for (int load_option = -1; load_option <= 1; ++load_option) {
        SCOPED_TRACE(process_option);
        SCOPED_TRACE(value == nullptr ? "no compile spec" : value);
        SCOPED_TRACE(load_option);
        ASSERT_EQ(set_graphs(process_option), Error::Ok);
        CompileSpec spec{kGraphOption, {value, 1}};
        BackendOption option_value = option(kGraphOption, load_option == 1);
        LoadedGraphEngine engine;
        ASSERT_EQ(
            engine.load(
                blob_,
                value == nullptr ? ::executorch::runtime::ArrayRef<CompileSpec>{}
                                 : ::executorch::runtime::ArrayRef<CompileSpec>(&spec, 1),
                load_option < 0 ? Span<const BackendOption>{} : Span<const BackendOption>(&option_value, 1)),
            Error::Ok);
        bool expected = process_option;
        if (value != nullptr) {
          expected = process_option && value[0] == '1';
        }
        if (load_option >= 0) {
          expected = load_option == 1;
        }
        EXPECT_EQ(engine.handle()->execution_graph != nullptr, expected);
      }
    }
  }
  BackendOption wrong_type = option(kGraphOption, true);
  wrong_type.value = 1;
  LoadedGraphEngine engine;
  EXPECT_EQ(engine.load(blob_, {}, {&wrong_type, 1}), Error::InvalidArgument);
}

TEST_F(ExecutionGraphReplayTest, CompileSpecRespectsProcessRefusalAndRejectsInvalidValues) {
  for (bool process_option : {false, true}) {
    for (const char* value : {"0", "1"}) {
      SCOPED_TRACE(process_option);
      SCOPED_TRACE(value);
      ASSERT_EQ(set_graphs(process_option), Error::Ok);
      CompileSpec spec{kGraphOption, {const_cast<char*>(value), 1}};
      LoadedGraphEngine engine;
      ASSERT_EQ(engine.load(blob_, {&spec, 1}), Error::Ok);
      EXPECT_EQ(engine.handle()->execution_graph != nullptr, process_option && value[0] == '1');
    }
  }
  for (const char* value : {"", "2", "10", "true"}) {
    SCOPED_TRACE(value);
    CompileSpec spec{kGraphOption, {const_cast<char*>(value), std::strlen(value)}};
    LoadedGraphEngine engine;
    EXPECT_EQ(engine.load(blob_, {&spec, 1}), Error::InvalidProgram);
  }
  char on[] = "1";
  CompileSpec twice[] = {{kGraphOption, {on, 1}}, {kGraphOption, {on, 1}}};
  LoadedGraphEngine engine;
  EXPECT_EQ(engine.load(blob_, {twice, 2}), Error::InvalidProgram);
}

TEST_F(ExecutionGraphReplayTest, ExplicitProcessRefusalPreventsSavedReplay) {
  ASSERT_EQ(set_graphs(false), Error::Ok);
  char on[] = "1";
  CompileSpec spec{kGraphOption, {on, 1}};
  LoadedGraphEngine engine;
  ASSERT_EQ(engine.load(blob_, {&spec, 1}), Error::Ok);
  EXPECT_EQ(engine.handle()->execution_graph, nullptr);
  std::vector<float> input(16, 3.0f), output(16, -1.0f);
  CudaCalls calls;
  for (int i = 0; i < 4; ++i) {
    ASSERT_EQ(engine.run(input.data(), output.data(), 2, 8), Error::Ok);
    EXPECT_EQ(output, std::vector<float>(16, 7.0f));
  }
  EXPECT_TRUE(calls.captures.empty());
  EXPECT_TRUE(calls.launches.empty());
}

TEST_F(ExecutionGraphReplayTest, CapturesSecondCallAndReplaysOnTheCallerAddresses) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  CudaCalls calls;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  EXPECT_TRUE(calls.launches.empty());
  EXPECT_FALSE(graph.is_captured());
  for (int i = 0; i < 7; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 2, streams_[0]));
    ASSERT_TRUE(graph.is_captured());
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[0]);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[0]);
  }
  ASSERT_EQ(calls.captures.size(), 1u);
  EXPECT_NE(calls.captures.front(), streams_[0]);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(7, streams_[0]));
  EXPECT_EQ(calls.malloc_calls, 0u);
  // Replay copies nothing: the only copies here are compare()'s own uploads, one per call.
  CudaCalls replay;
  ASSERT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, streams_[0]), Error::Ok);
  EXPECT_EQ(replay.memcpy_calls, 0u);
  EXPECT_EQ(replay.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

// One move records again on the new addresses. A second move switches to the graph's own buffers,
// which every later call copies through, so a caller alternating two sets of buffers still replays.
TEST_F(ExecutionGraphReplayTest, MovedAddressesRecordAgainOnceThenCopyThroughOwnBuffers) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  const int slots[] = {0, 0, 1, 0, 1, 0, 1, 0};
  const bool captured[] = {false, true, true, true, true, true, true, true};
  CudaCalls calls;
  for (int i = 0; i < 8; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, slots[i], 2, 8, i, streams_[0]));
    EXPECT_EQ(graph.is_captured(), captured[i]);
    const bool staged = i >= 3;
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0") == inputs_[slots[i]], !staged);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0") == outputs_[slots[i]], !staged);
  }
  EXPECT_EQ(calls.captures.size(), 3u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(7, streams_[0]));
  EXPECT_EQ(calls.malloc_calls, 2u);
  // Each staged replay copies its input in and its output out.
  CudaCalls replay;
  ASSERT_EQ(graph.run(inputs_[1], outputs_[1], 2, 8, streams_[0]), Error::Ok);
  EXPECT_EQ(replay.memcpy_calls, 2u);
  EXPECT_EQ(replay.launches.size(), 1u);
}

// Buffers that move on every call never repeat the previous call's, so each set of shapes records once, on the
// graph's own buffers, as for buffers that stay put.
TEST_F(ExecutionGraphReplayTest, BuffersThatMoveOnEveryCallRecordOnceForEachSetOfShapes) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  const SizesType shapes[][2] = {{2, 8}, {4, 4}, {2, 8}, {4, 4}};
  std::vector<size_t> captures;
  int call = 0;
  for (const auto& shape : shapes) {
    CudaCalls calls;
    // Three calls for each set of shapes, on two sets of buffers in turn.
    for (int i = 0; i < 3; ++i, ++call) {
      SCOPED_TRACE(call);
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, call % 2, shape[0], shape[1], call, streams_[0]));
    }
    captures.push_back(calls.captures.size());
  }
  EXPECT_EQ(captures, std::vector<size_t>({1, 1, 1, 1}));
}

// Buffers that stay put after one call on other buffers record again on the caller's addresses, release the
// graph's own buffers, and replay with no copies.
TEST_F(ExecutionGraphReplayTest, BuffersThatSettleReplayOnThemAndReleaseOwnBuffers) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  CudaCalls calls;
  for (int i = 0; i < 5; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i == 0 ? 0 : 1, 2, 8, i, streams_[0]));
  }
  EXPECT_EQ(calls.captures.size(), 2u);
  EXPECT_EQ(calls.malloc_calls, 2u);
  EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, streams_[0]));
  EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[1]);
  EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[1]);
  CudaCalls replay;
  ASSERT_EQ(graph.run(inputs_[1], outputs_[1], 2, 8, streams_[0]), Error::Ok);
  EXPECT_EQ(replay.memcpy_calls, 0u);
  EXPECT_EQ(replay.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

// Buffers that settle under new shapes record on the caller's addresses, and the graph's own buffers from the
// call before go then, not when the engine is destroyed.
TEST_F(ExecutionGraphReplayTest, BuffersThatSettleUnderNewShapesReleaseOwnBuffers) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  CudaCalls calls;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 0, streams_[0]));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 1, 2, 8, 1, streams_[0]));
  EXPECT_EQ(calls.malloc_calls, 2u);
  for (int i = 2; i < 5; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 1, 4, 8, i, streams_[0]));
  }
  EXPECT_EQ(calls.captures.size(), 2u);
  EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, streams_[0]));
  EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[1]);
}

// The graph's own buffers are not needed for buffers that settle, so a failure to allocate them still leaves a
// recording on the caller's addresses that replays with no copies.
TEST_F(ExecutionGraphReplayTest, BuffersThatSettleReplayOnThemWhenOwnBuffersFailToAllocate) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  CudaCalls calls;
  calls.fail_malloc_call = 1;
  for (int i = 0; i < 5; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i == 0 ? 0 : 1, 2, 8, i, streams_[0]));
  }
  EXPECT_EQ(calls.malloc_calls, 1u);
  EXPECT_EQ(calls.captures.size(), 1u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(3, streams_[0]));
  EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[1]);
  EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[1]);
}

// Without buffers of its own, a graph still records only on caller buffers that repeat, so buffers that keep
// moving after the allocation fails run plainly instead of recording on every call.
TEST_F(ExecutionGraphReplayTest, BuffersThatKeepMovingWhenOwnBuffersFailToAllocateRunPlainly) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  CudaCalls calls;
  calls.fail_malloc_call = 1;
  for (int i = 0; i < 10; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, streams_[0]));
  }
  EXPECT_EQ(calls.malloc_calls, 1u);
  EXPECT_TRUE(calls.captures.empty());
  EXPECT_TRUE(calls.launches.empty());
  for (int i = 10; i < 13; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 1, 2, 8, i, streams_[0]));
  }
  EXPECT_EQ(calls.captures.size(), 1u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(3, streams_[0]));
}

// The output moves while the input stays put. Replaying on the recorded output would leave the new one unwritten.
TEST_F(ExecutionGraphReplayTest, OutputOnlyMovesWriteTheNewOutput) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  std::vector<float> input(16);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  ASSERT_EQ(cudaMemcpy(inputs_[0], input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice), cudaSuccess);
  const int slots[] = {0, 0, 1, 0, 1, 0, 1, 0};
  for (int i = 0; i < 8; ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(cudaMemsetAsync(outputs_[slots[i]], 0xff, kMaxBytes, streams_[0]), cudaSuccess);
    ASSERT_EQ(graph.run(inputs_[0], outputs_[slots[i]], 2, 8, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    std::vector<float> output(input.size());
    ASSERT_EQ(
        cudaMemcpy(output.data(), outputs_[slots[i]], sizeof(float) * output.size(), cudaMemcpyDeviceToHost),
        cudaSuccess);
    for (size_t j = 0; j < output.size(); ++j) {
      ASSERT_FLOAT_EQ(output[j], input[j] * 2.0f + 1.0f) << "element " << j;
    }
  }
  EXPECT_TRUE(graph.is_captured());
}

// With or without graphs, a call gives TensorRT a shape or an address again only when it changed. What the
// test sets on the TensorRT context behind the delegate's back stays there exactly when the call is skipped.
TEST_F(ExecutionGraphReplayTest, UnchangedShapesAndAddressesAreNotSetAgain) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  for (LoadedGraphEngine* engine : {&plain, &graph}) {
    nvinfer1::IExecutionContext& context = *engine->handle()->exec_ctx;
    // compare() gives the plain engine its own output buffer.
    void* const output = engine == &plain ? reference_output_ : outputs_[0];
    // As many elements as 2x8, so the output tensor takes the shape the context reports.
    ASSERT_TRUE(context.setInputShape("input_0", nvinfer1::Dims2{4, 4}));
    ASSERT_TRUE(context.setTensorAddress("input_0", inputs_[1]));
    ASSERT_TRUE(context.setTensorAddress("output_0", outputs_[1]));
    SizesType input_sizes[] = {2, 8};
    SizesType output_sizes[] = {2, 8};
    DimOrderType dim_order[] = {0, 1};
    StridesType input_strides[] = {8, 1};
    StridesType output_strides[] = {8, 1};
    TensorImpl input_impl(
        ScalarType::Float, 2, input_sizes, inputs_[0], dim_order, input_strides, TensorShapeDynamism::DYNAMIC_BOUND);
    TensorImpl output_impl(
        ScalarType::Float, 2, output_sizes, output, dim_order, output_strides, TensorShapeDynamism::DYNAMIC_BOUND);
    EValue input_value{Tensor(&input_impl)};
    EValue output_value{Tensor(&output_impl)};
    EValue* args[] = {&input_value, &output_value};
    ASSERT_EQ(engine->run({args, 2}, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    const nvinfer1::Dims shape = context.getTensorShape("input_0");
    EXPECT_EQ(shape.nbDims, 2);
    EXPECT_EQ(shape.d[0], 4);
    EXPECT_EQ(shape.d[1], 4);
    EXPECT_EQ(context.getTensorAddress("input_0"), inputs_[1]);
    EXPECT_EQ(context.getTensorAddress("output_0"), outputs_[1]);
    // Back to what the delegate holds, so the calls below run as usual.
    ASSERT_TRUE(context.setInputShape("input_0", nvinfer1::Dims2{2, 8}));
    ASSERT_TRUE(context.setTensorAddress("input_0", inputs_[0]));
    ASSERT_TRUE(context.setTensorAddress("output_0", output));
  }
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 2, streams_[0]));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 4, 4, 3, streams_[0]));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 4, streams_[0]));
}

// The layout check reads strides and needs no dim order, so a column-major tensor with strides but no
// dim order is copied into row-major order, with or without replay.
TEST_F(ExecutionGraphReplayTest, StridesWithoutADimOrderAreReadByTheirStrides) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  std::vector<float> storage(16);
  for (size_t i = 0; i < storage.size(); ++i) {
    storage[i] = static_cast<float>(i);
  }
  ASSERT_EQ(
      cudaMemcpy(inputs_[0], storage.data(), sizeof(float) * storage.size(), cudaMemcpyHostToDevice), cudaSuccess);
  for (LoadedGraphEngine* engine : {&plain, &graph}) {
    for (int i = 0; i < 3; ++i) {
      SCOPED_TRACE(i);
      SizesType input_sizes[] = {2, 8};
      SizesType output_sizes[] = {2, 8};
      StridesType input_strides[] = {1, 2};
      TensorImpl input_impl(ScalarType::Float, 2, input_sizes, inputs_[0], nullptr, input_strides);
      TensorImpl output_impl(ScalarType::Float, 2, output_sizes, outputs_[0]);
      EValue input_value{Tensor(&input_impl)};
      EValue output_value{Tensor(&output_impl)};
      EValue* args[] = {&input_value, &output_value};
      ASSERT_EQ(cudaMemsetAsync(outputs_[0], 0xff, kMaxBytes, streams_[0]), cudaSuccess);
      ASSERT_EQ(engine->run({args, 2}, streams_[0]), Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
      std::vector<float> output(storage.size());
      ASSERT_EQ(
          cudaMemcpy(output.data(), outputs_[0], sizeof(float) * output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
      // Row r, column c of the column-major input is storage[c * 2 + r].
      for (size_t j = 0; j < output.size(); ++j) {
        const size_t r = j / 8, c = j % 8;
        ASSERT_FLOAT_EQ(output[j], storage[c * 2 + r] * 2.0f + 1.0f) << "element " << j;
      }
    }
  }
  EXPECT_TRUE(graph.is_captured());
}

// 16 floats of host memory at a chosen offset from a 256-byte boundary. Where the GPU reads host memory
// through the host page tables, an aligned host buffer is bound as it is, while an unaligned one always
// goes through the delegate's own device buffer, so the offset decides which path a test takes.
struct HostFloats {
  explicit HostFloats(size_t offset_bytes)
      : base(static_cast<float*>(std::aligned_alloc(256, 512))), data(base + offset_bytes / sizeof(float)) {}
  ~HostFloats() {
    std::free(base);
  }
  HostFloats(const HostFloats&) = delete;
  HostFloats& operator=(const HostFloats&) = delete;

  float* base;
  float* data;
};

TEST_F(ExecutionGraphReplayTest, NoCallerStreamReplaysHostPointers) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  // Bound as it is where the GPU reads host memory, and copied through the delegate's own device buffer
  // elsewhere.
  const bool direct = graph.handle()->pageable_host_access;
  HostFloats input(0), output(0);
  std::vector<float> expected(16);
  size_t launches = 0;
  size_t captures = 0;
  for (int i = 0; i < 6; ++i) {
    SCOPED_TRACE(i);
    for (size_t j = 0; j < 16; ++j) {
      input.data[j] = static_cast<float>(i * 16 + j);
      output.data[j] = -1000.0f;
    }
    ASSERT_FALSE(getCallerStream().has_value());
    ASSERT_EQ(plain.run(input.data, expected.data(), 2, 8), Error::Ok);
    CudaCalls calls;
    ASSERT_EQ(graph.run(input.data, output.data, 2, 8), Error::Ok);
    EXPECT_FALSE(getCallerStream().has_value());
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_FLOAT_EQ(output.data[j], expected[j]);
      EXPECT_FLOAT_EQ(output.data[j], input.data[j] * 2.0f + 1.0f);
    }
    ASSERT_FALSE(calls.operations.empty());
    EXPECT_EQ(calls.operations.back().stream, cudaStreamPerThread);
    EXPECT_EQ(graph.is_captured(), i >= 1);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0") == input.data, direct);
    captures += calls.captures.size();
    launches += calls.launches.size();
  }
  EXPECT_EQ(captures, 1u);
  EXPECT_EQ(launches, 5u);
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamReplaysDevicePointers) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  size_t launches = 0;
  size_t captures = 0;
  for (int i = 0; i < 6; ++i) {
    SCOPED_TRACE(i);
    CudaCalls calls;
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, i));
    ASSERT_FALSE(calls.operations.empty());
    EXPECT_EQ(calls.operations.back().stream, cudaStreamPerThread);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[0]);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[0]);
    EXPECT_EQ(graph.is_captured(), i >= 1);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(i == 0 ? 0 : 1, cudaStreamPerThread));
    captures += calls.captures.size();
    launches += calls.launches.size();
  }
  EXPECT_EQ(captures, 1u);
  EXPECT_EQ(launches, 5u);
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamWaitsForReplayBeforeReturning) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, i));
  }
  struct Completion {
    std::atomic<bool> release{false};
    std::atomic<bool> completed{false};
  } completion;
  cudaStream_t replay_stream = nullptr;
  bool pending_at_sync = false;
  ASSERT_EQ(cudaMemsetAsync(outputs_[0], 0xff, kMaxBytes, cudaStreamPerThread), cudaSuccess);
  CudaCalls calls;
  calls.after_launch = [&](cudaStream_t stream) {
    replay_stream = stream;
    EXPECT_EQ(
        cudaLaunchHostFunc(
            stream,
            [](void* data) {
              auto& state = *static_cast<Completion*>(data);
              while (!state.release.load()) {
                std::this_thread::yield();
              }
              state.completed.store(true);
            },
            &completion),
        cudaSuccess);
  };
  calls.before_synchronize = [&](cudaStream_t stream) {
    EXPECT_EQ(stream, replay_stream);
    pending_at_sync = !completion.completed.load();
    completion.release.store(true);
  };
  ASSERT_FALSE(getCallerStream().has_value());
  const auto result = graph.run(inputs_[0], outputs_[0], 2, 8);
  const bool completed_on_return = completion.completed.load();
  // Release even when execute forgot to drain, so a failing assertion cannot strand the callback.
  completion.release.store(true);
  calls.before_synchronize = nullptr;
  EXPECT_EQ(result, Error::Ok);
  EXPECT_TRUE(pending_at_sync);
  EXPECT_TRUE(completed_on_return);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, cudaStreamPerThread));
  ASSERT_EQ(cudaStreamSynchronize(replay_stream), cudaSuccess);
  std::vector<float> output(16);
  ASSERT_EQ(cudaMemcpy(output.data(), outputs_[0], 64, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>((16 + i) * 2 + 1));
  }
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamAllocationFailureFallsBackAndSuppressesRetries) {
  for (size_t failed_allocation : {1u, 2u}) {
    for (bool growth : {false, true}) {
      SCOPED_TRACE(failed_allocation);
      SCOPED_TRACE(growth);
      LoadedGraphEngine graph;
      ASSERT_NO_FATAL_FAILURE(load_graph(graph));
      if (growth) {
        // Staged under other shapes, so the first call of the loop, a warm-up, grows the buffers.
        ASSERT_NO_FATAL_FAILURE(stage_default(graph, 2, 8));
      } else {
        // A warm-up, a recording and the first move. The second move, which allocates the buffers, is the
        // first call of the loop after this one.
        for (int slot : {0, 0, 1}) {
          ASSERT_NO_FATAL_FAILURE(compare_default(graph, slot, 8, 32, slot));
        }
      }
      CudaCalls calls;
      calls.fail_malloc_call = failed_allocation;
      for (int i = 0; i < 5; ++i) {
        ASSERT_NO_FATAL_FAILURE(compare_default(graph, i % 2, 8, 32, i));
        EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
        EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
        EXPECT_FALSE(graph.is_captured());
      }
      EXPECT_EQ(calls.malloc_calls, failed_allocation);
      EXPECT_TRUE(calls.captures.empty());
      EXPECT_TRUE(calls.launches.empty());
      EXPECT_EQ(calls.free_calls, 0u);
      EXPECT_EQ(calls.retirements.size(), (growth ? 2 : 0) + failed_allocation - 1);
    }
  }
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamCaptureFailureFallsBackAndRecovers) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, 0));
  {
    CudaCalls calls;
    calls.fail_captures = 3;
    // The same buffers until recording gives up, so every recording goes on the caller's addresses.
    const int slots[] = {0, 0, 0, 1, 0, 1};
    for (int i = 0; i < 6; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare_default(graph, slots[i], 2, 8, i + 1));
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[slots[i]]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[slots[i]]);
      EXPECT_FALSE(graph.is_captured());
    }
    ASSERT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
    EXPECT_TRUE(calls.retirements.empty());
    EXPECT_EQ(calls.malloc_calls, 0u);
    EXPECT_EQ(calls.free_calls, 0u);
  }
  CudaCalls recovery;
  // The new shape re-arms replay: warm-up, then capture and launch, then replays.
  for (int i = 0; i < 4; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 4, 4, i + 6));
  }
  ASSERT_EQ(recovery.captures.size(), 1u);
  EXPECT_EQ(recovery.launches, std::vector<cudaStream_t>(3, cudaStreamPerThread));
}

TEST_F(ExecutionGraphReplayTest, AFailedRecordingIsRetriedOnTheNextCall) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  CudaCalls calls;
  calls.fail_captures = 2;
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 2, streams_[0]));
    EXPECT_FALSE(graph.is_captured());
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[0]);
  }
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 4, streams_[0]));
    EXPECT_TRUE(graph.is_captured());
  }
  EXPECT_EQ(calls.captures.size(), 3u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(3, streams_[0]));
  EXPECT_EQ(calls.malloc_calls, 0u);
  EXPECT_TRUE(calls.retirements.empty());
}

TEST_F(ExecutionGraphReplayTest, FirstNoStreamCallOnAFreshThreadReplays) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, i));
  }
  ASSERT_TRUE(graph.is_captured());
  std::vector<float> input(16);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i) - 3.0f;
  }
  ASSERT_EQ(cudaMemcpy(inputs_[0], input.data(), 64, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(outputs_[0], 0xff, kMaxBytes), cudaSuccess);
  size_t mallocs = 0, launches = 0, captures = 0;
  Error result = Error::Internal;
  // No CUDA call may run on this thread before execute, so it starts with no current context.
  std::thread fresh([&] {
    CudaCalls calls;
    result = graph.run(inputs_[0], outputs_[0], 2, 8);
    captures = calls.captures.size();
    mallocs = calls.malloc_calls;
    launches = calls.launches.size();
  });
  fresh.join();
  EXPECT_EQ(result, Error::Ok);
  EXPECT_EQ(mallocs, 0u);
  EXPECT_EQ(captures, 0u);
  EXPECT_EQ(launches, 1u);
  EXPECT_TRUE(graph.is_captured());
  std::vector<float> output(16);
  ASSERT_EQ(cudaMemcpy(output.data(), outputs_[0], 64, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i], input[i] * 2.0f + 1.0f);
  }
}

TEST_F(ExecutionGraphReplayTest, DestroyFreesReplayBuffers) {
  auto graph = std::make_unique<LoadedGraphEngine>();
  ASSERT_NO_FATAL_FAILURE(load_graph(*graph));
  ASSERT_NO_FATAL_FAILURE(stage_default(*graph, 2, 8));
  CudaCalls calls;
  graph.reset();
  EXPECT_EQ(calls.free_calls, 2u);
}

TEST_F(ExecutionGraphReplayTest, EnqueueRefusedDuringCaptureFallsBackWithFreshOutputs) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  {
    CudaCalls calls;
    calls.during_capture = [&](cudaStream_t) {
      // TensorRT refuses an enqueue with an input that has no address.
      EXPECT_TRUE(graph.handle()->exec_ctx->setTensorAddress("input_0", nullptr));
    };
    for (int i = 0; i < 4; ++i) {
      SCOPED_TRACE(i);
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i + 2, streams_[0]));
      EXPECT_FALSE(graph.is_captured());
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
    }
    EXPECT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
  }
  CudaCalls recovery;
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 4, 4, i + 6, streams_[0]));
  }
  EXPECT_EQ(recovery.captures.size(), 1u);
  EXPECT_EQ(recovery.launches, std::vector<cudaStream_t>(2, streams_[0]));
}

// Whatever made TensorRT refuse may also have changed the context's shapes, so they are set again, and
// this call and the next one with the same shapes run on the right ones.
TEST_F(ExecutionGraphReplayTest, ARefusedCaptureSetsTheShapesAgain) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  {
    CudaCalls calls;
    calls.during_capture = [&](cudaStream_t) {
      EXPECT_TRUE(graph.handle()->exec_ctx->setTensorAddress("input_0", nullptr));
      EXPECT_TRUE(graph.handle()->exec_ctx->setInputShape("input_0", nvinfer1::Dims2{1, 8}));
    };
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 2, streams_[0]));
    EXPECT_EQ(calls.captures.size(), 1u);
    EXPECT_FALSE(graph.is_captured());
  }
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 3, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamLaunchErrorsDrainAndRecover) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, i));
  }
  for (cudaError_t error : {cudaErrorMemoryAllocation, cudaErrorInvalidValue}) {
    CudaCalls calls;
    calls.launch_error = error;
    EXPECT_EQ(
        graph.run(inputs_[0], outputs_[0], 2, 8),
        error == cudaErrorMemoryAllocation ? Error::MemoryAllocationFailed : Error::Internal);
    ASSERT_EQ(calls.launches.size(), 1u);
    ASSERT_FALSE(calls.operations.empty());
    EXPECT_EQ(calls.operations.back().kind, CudaCallKind::Synchronize);
    EXPECT_EQ(calls.operations.back().stream, calls.launches.front());
  }
  CudaCalls recovery;
  ASSERT_NO_FATAL_FAILURE(compare_default(graph, 1, 2, 8, 9));
  EXPECT_EQ(recovery.launches.size(), 1u);
}

TEST_F(ExecutionGraphTest, NoCallerStreamIneligibleEnginesStayPlain) {
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode);
    ASSERT_EQ(set_graphs(mode != 0), Error::Ok);
    LoadedGraphEngine graph;
    {
      CudaCalls initialization;
      initialization.disable_memory_pools = mode == 1;
      ASSERT_EQ(graph.load(mode == 2 ? alias_blob_ : blob_), Error::Ok);
    }
    EXPECT_EQ(graph.handle()->execution_graph, nullptr);
    CudaCalls calls;
    for (int i = 0; i < 4; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare_default(graph, i % 2, 2, 8, i));
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), mode == 2 ? inputs_[i % 2] : outputs_[i % 2]);
    }
    EXPECT_EQ(calls.malloc_calls, 0u);
    EXPECT_TRUE(calls.captures.empty());
    EXPECT_TRUE(calls.launches.empty());
  }
}

TEST_F(ExecutionGraphReplayTest, ShapeChangesResetAndRecaptureIncludingEqualByteSizes) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  const SizesType shapes[][2] = {{2, 8}, {4, 4}, {2, 4}, {8, 32}, {1, 2}, {2, 8}};
  int seed = 0;
  for (const auto& shape : shapes) {
    SCOPED_TRACE(++seed);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, shape[0], shape[1], seed, streams_[0]));
    ASSERT_FALSE(graph.is_captured());
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 1, shape[0], shape[1], seed + 1, streams_[0]));
    ASSERT_TRUE(graph.is_captured());
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, shape[0], shape[1], seed + 2, streams_[0]));
    ASSERT_TRUE(graph.is_captured());
  }
}

// A graph replays the kernels of the profile it was captured under, so switching
// profiles at an unchanged shape must drop it rather than replay the old profile.
TEST_F(ExecutionGraphReplayTest, ProfileSwitchAtTheSameShapeResetsAndRecaptures) {
  const auto two_profiles = build_blob(false, 2);
  ASSERT_FALSE(two_profiles.empty());
  LoadedGraphEngine graph;
  ASSERT_EQ(graph.load(two_profiles), Error::Ok);
  ASSERT_NE(graph.handle()->execution_graph, nullptr);
  std::vector<float> input(16), output(16);
  auto run_on = [&](int32_t profile, float base) {
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = base + static_cast<float>(i);
    }
    std::fill(output.begin(), output.end(), -1.0f);
    OptimizationProfileGuard guard{profile};
    ASSERT_EQ(graph.run(input.data(), output.data(), 2, 8, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    for (size_t i = 0; i < input.size(); ++i) {
      ASSERT_FLOAT_EQ(output[i], input[i] * 2.0f + 1.0f) << "profile " << profile << " element " << i;
    }
    EXPECT_EQ(graph.handle()->profiles.active, profile);
  };
  ASSERT_NO_FATAL_FAILURE(run_on(0, 0.0f));
  ASSERT_NO_FATAL_FAILURE(run_on(0, 10.0f));
  ASSERT_TRUE(graph.is_captured());
  ASSERT_NO_FATAL_FAILURE(run_on(1, 20.0f));
  EXPECT_FALSE(graph.is_captured()) << "a graph captured under profile 0 survived the switch to profile 1";
  ASSERT_NO_FATAL_FAILURE(run_on(1, 30.0f));
  EXPECT_TRUE(graph.is_captured());
  ASSERT_NO_FATAL_FAILURE(run_on(1, 40.0f));
  EXPECT_TRUE(graph.is_captured());
}

// A profile switch leaves the context's shapes and addresses unknown to the delegate, so the next call sets
// each one again. What the test sets on the TensorRT context behind the delegate's back must not survive it.
TEST_F(ExecutionGraphTest, AProfileSwitchSetsShapesAndAddressesAgain) {
  const auto two_profiles = build_blob(false, 2);
  ASSERT_FALSE(two_profiles.empty());
  ASSERT_EQ(set_graphs(false), Error::Ok);
  LoadedGraphEngine engine;
  ASSERT_EQ(engine.load(two_profiles), Error::Ok);
  ASSERT_EQ(engine.handle()->execution_graph, nullptr);
  std::vector<float> input(16);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }
  ASSERT_EQ(cudaMemcpy(inputs_[0], input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice), cudaSuccess);
  {
    OptimizationProfileGuard guard{0};
    ASSERT_EQ(engine.run(inputs_[0], outputs_[0], 2, 8, streams_[0]), Error::Ok);
  }
  nvinfer1::IExecutionContext& context = *engine.handle()->exec_ctx;
  ASSERT_TRUE(context.setInputShape("input_0", nvinfer1::Dims2{4, 4}));
  ASSERT_TRUE(context.setTensorAddress("input_0", inputs_[1]));
  ASSERT_TRUE(context.setTensorAddress("output_0", outputs_[1]));
  ASSERT_EQ(cudaMemset(outputs_[0], 0xff, kMaxBytes), cudaSuccess);
  {
    OptimizationProfileGuard guard{1};
    ASSERT_EQ(engine.run(inputs_[0], outputs_[0], 2, 8, streams_[0]), Error::Ok);
  }
  ASSERT_EQ(engine.handle()->profiles.active, 1);
  const nvinfer1::Dims shape = context.getTensorShape("input_0");
  ASSERT_EQ(shape.nbDims, 2);
  EXPECT_EQ(shape.d[0], 2);
  EXPECT_EQ(shape.d[1], 8);
  EXPECT_EQ(context.getTensorAddress("input_0"), inputs_[0]);
  EXPECT_EQ(context.getTensorAddress("output_0"), outputs_[0]);
  std::vector<float> output(input.size());
  ASSERT_EQ(cudaMemcpy(output.data(), outputs_[0], sizeof(float) * output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < output.size(); ++i) {
    ASSERT_FLOAT_EQ(output[i], input[i] * 2.0f + 1.0f) << "element " << i;
  }
}

TEST_F(ExecutionGraphReplayTest, StreamChangesKeepTheGraph) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  ASSERT_FALSE(graph.is_captured());
  CudaCalls calls;
  int seed = 2;
  for (cudaStream_t stream : {streams_[0], streams_[1], streams_[0], streams_[1]}) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed, stream));
    ++seed;
    ASSERT_TRUE(graph.is_captured());
  }
  EXPECT_EQ(calls.captures.size(), 1u);
  EXPECT_EQ(calls.malloc_calls, 0u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>({streams_[0], streams_[1], streams_[0], streams_[1]}));
}

TEST_F(ExecutionGraphReplayTest, BlockingAndDefaultStreamsReplay) {
  cudaStream_t blocking = nullptr;
  ASSERT_EQ(cudaStreamCreate(&blocking), cudaSuccess);
  for (cudaStream_t stream : {blocking, cudaStreamLegacy, cudaStreamPerThread, static_cast<cudaStream_t>(nullptr)}) {
    LoadedGraphEngine plain, graph;
    ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
    CudaCalls calls;
    for (int i = 0; i < 6; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, stream));
      EXPECT_EQ(graph.is_captured(), i >= 1);
    }
    ASSERT_EQ(calls.captures.size(), 1u);
    EXPECT_NE(calls.captures.front(), stream);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(5, stream));
  }
  EXPECT_EQ(cudaStreamDestroy(blocking), cudaSuccess);
}

// Counts the errors TensorRT logs, so a test can check that its steps logged none.
std::atomic<int> tensorrt_errors{0};

void count_tensorrt_errors(
    et_timestamp_t,
    et_pal_log_level_t level,
    const char*,
    const char*,
    size_t,
    const char* message,
    size_t length) {
  std::fprintf(stderr, "%c %.*s\n", static_cast<char>(level), static_cast<int>(length), message);
  if (std::string_view(message, length).rfind("TensorRT: ", 0) == 0) {
    tensorrt_errors.fetch_add(1);
  }
}

// Green contexts need a driver and a GPU that support them. Tests that need one skip where creation is
// refused, as the multi-GPU tests skip without a second device.
bool green_contexts_supported(int device = 0) {
  const auto get_resource = driver_entry<PFN_cuDeviceGetDevResource_v12040>("cuDeviceGetDevResource", 12040);
  const auto generate_desc = driver_entry<PFN_cuDevResourceGenerateDesc_v12040>("cuDevResourceGenerateDesc", 12040);
  const auto create_green = driver_entry<PFN_cuGreenCtxCreate_v12040>("cuGreenCtxCreate", 12040);
  const auto destroy = driver_entry<PFN_cuGreenCtxDestroy_v12040>("cuGreenCtxDestroy", 12040);
  if (get_resource == nullptr || generate_desc == nullptr || create_green == nullptr || destroy == nullptr) {
    return false;
  }
  CUdevResource resource{};
  CUdevResourceDesc descriptor{};
  CUgreenCtx context = nullptr;
  if (get_resource(device, &resource, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS ||
      generate_desc(&descriptor, &resource, 1) != CUDA_SUCCESS ||
      create_green(&context, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM) != CUDA_SUCCESS) {
    cudaGetLastError();
    return false;
  }
  return destroy(context) == CUDA_SUCCESS;
}

// Asks the driver rather than the pool, so a pool that wrongly makes no handoff flag fails the flag tests
// instead of skipping them.
bool device_can_wait_on_a_flag(int ordinal = 0) {
  const auto get_device = driver_entry<PFN_cuDeviceGet_v2000>("cuDeviceGet", 2000);
  const auto get_attribute = driver_entry<PFN_cuDeviceGetAttribute_v2000>("cuDeviceGetAttribute", 2000);
  CUdevice device = 0;
  int supported = 0;
  return get_device != nullptr && get_attribute != nullptr && get_device(&device, ordinal) == CUDA_SUCCESS &&
      get_attribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, device) == CUDA_SUCCESS &&
      supported != 0;
}

// A non-blocking stream on a green context.
struct GreenStream {
  ~GreenStream() {
    if (stream) {
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
    if (context) {
      EXPECT_EQ(destroy(context), CUDA_SUCCESS);
    }
  }

  void create(int device = 0) {
    const auto get_resource = driver_entry<PFN_cuDeviceGetDevResource_v12040>("cuDeviceGetDevResource", 12040);
    const auto generate_desc = driver_entry<PFN_cuDevResourceGenerateDesc_v12040>("cuDevResourceGenerateDesc", 12040);
    const auto create_green = driver_entry<PFN_cuGreenCtxCreate_v12040>("cuGreenCtxCreate", 12040);
    destroy = driver_entry<PFN_cuGreenCtxDestroy_v12040>("cuGreenCtxDestroy", 12040);
    const auto create_stream = driver_entry<PFN_cuGreenCtxStreamCreate_v12050>("cuGreenCtxStreamCreate", 12050);
    const auto get_context = driver_entry<PFN_cuStreamGetCtx_v12050>("cuStreamGetCtx", 12050);
    ASSERT_NE(get_resource, nullptr);
    ASSERT_NE(generate_desc, nullptr);
    ASSERT_NE(create_green, nullptr);
    ASSERT_NE(destroy, nullptr);
    ASSERT_NE(create_stream, nullptr);
    ASSERT_NE(get_context, nullptr);
    CUdevResource resource{};
    CUdevResourceDesc descriptor{};
    ASSERT_EQ(get_resource(device, &resource, CU_DEV_RESOURCE_TYPE_SM), CUDA_SUCCESS);
    ASSERT_EQ(generate_desc(&descriptor, &resource, 1), CUDA_SUCCESS);
    ASSERT_EQ(create_green(&context, descriptor, device, CU_GREEN_CTX_DEFAULT_STREAM), CUDA_SUCCESS);
    ASSERT_EQ(create_stream(&stream, context, CU_STREAM_NON_BLOCKING, 0), CUDA_SUCCESS);
    CUcontext owner = nullptr;
    CUgreenCtx observed_green = nullptr;
    ASSERT_EQ(get_context(stream, &owner, &observed_green), CUDA_SUCCESS);
    ASSERT_EQ(observed_green, context);
    ASSERT_NE(observed_green, nullptr);
    unsigned int flags = 0;
    ASSERT_EQ(cudaStreamGetFlags(stream, &flags), cudaSuccess);
    ASSERT_NE(flags & cudaStreamNonBlocking, 0u);
  }

  CUgreenCtx context = nullptr;
  cudaStream_t stream = nullptr;
  PFN_cuGreenCtxDestroy_v12040 destroy = nullptr;
};

// The context of each kernel node in a recorded graph, which is where that kernel runs on replay.
std::vector<CUcontext> kernel_contexts(cudaGraph_t graph) {
  const auto get_params = driver_entry<PFN_cuGraphKernelNodeGetParams_v12000>("cuGraphKernelNodeGetParams", 12000);
  if (get_params == nullptr) {
    ADD_FAILURE() << "cuGraphKernelNodeGetParams is missing";
    return {};
  }
  size_t count = 0;
  EXPECT_EQ(cudaGraphGetNodes(graph, nullptr, &count), cudaSuccess);
  std::vector<cudaGraphNode_t> nodes(count);
  EXPECT_EQ(cudaGraphGetNodes(graph, nodes.data(), &count), cudaSuccess);
  std::vector<CUcontext> contexts;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    if (cudaGraphNodeGetType(node, &type) == cudaSuccess && type == cudaGraphNodeTypeKernel) {
      CUDA_KERNEL_NODE_PARAMS params{};
      EXPECT_EQ(get_params(reinterpret_cast<CUgraphNode>(node), &params), CUDA_SUCCESS);
      contexts.push_back(params.ctx);
    }
  }
  return contexts;
}

// CUDA allows a green context handle on one CPU thread at a time, so two engines replaying on streams of one
// green context from two threads still use the handle one thread at a time.
TEST_F(ExecutionGraphReplayTest, EnginesOnOneGreenContextUseItOneThreadAtATime) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green;
  ASSERT_NO_FATAL_FAILURE(green.create());
  const auto create_stream = driver_entry<PFN_cuGreenCtxStreamCreate_v12050>("cuGreenCtxStreamCreate", 12050);
  ASSERT_NE(create_stream, nullptr);
  CUstream second_stream = nullptr;
  ASSERT_EQ(create_stream(&second_stream, green.context, CU_STREAM_NON_BLOCKING, 0), CUDA_SUCCESS);
  // Destroyed before the green context it belongs to.
  std::unique_ptr<CUstream_st, decltype(&cudaStreamDestroy)> second(second_stream, &cudaStreamDestroy);
  LoadedGraphEngine first_engine, second_engine;
  ASSERT_NO_FATAL_FAILURE(load_graph(first_engine));
  ASSERT_NO_FATAL_FAILURE(load_graph(second_engine));
  std::atomic<int> failures{0};
  const auto calls = [&](LoadedGraphEngine& engine, int slot, cudaStream_t stream) {
    for (int i = 0; i < 40; ++i) {
      if (engine.run(inputs_[slot], outputs_[slot], 2, 8, stream) != Error::Ok) {
        ++failures;
      }
    }
  };
  SlowGreenContextCalls slow;
  std::thread first([&] { calls(first_engine, 0, green.stream); });
  std::thread other([&] { calls(second_engine, 1, second.get()); });
  first.join();
  other.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(slow.most_at_once(), 1) << "two threads used one green context handle at the same time";
}

// Graph nodes run in the context they were recorded in, so a green context stream records its own
// graph there, and its replays stay on the green context's SMs. The plain stream's graph is kept.
TEST_F(ExecutionGraphReplayTest, GreenContextStreamRecordsInItsOwnContext) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green;
  ASSERT_NO_FATAL_FAILURE(green.create());
  const auto get_context = driver_entry<PFN_cuStreamGetCtx_v12050>("cuStreamGetCtx", 12050);
  const auto from_green = driver_entry<PFN_cuCtxFromGreenCtx_v12040>("cuCtxFromGreenCtx", 12040);
  ASSERT_NE(get_context, nullptr);
  ASSERT_NE(from_green, nullptr);
  CUcontext green_context = nullptr;
  ASSERT_EQ(from_green(&green_context, green.context), CUDA_SUCCESS);

  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  CUcontext owner = nullptr;
  CUgreenCtx capture_green = nullptr;
  {
    CudaCalls calls;
    std::vector<CUcontext> contexts;
    // The capture stream is destroyed right after the recording, so it is asked about before that.
    calls.after_capture = [&](cudaGraph_t recorded) {
      contexts = kernel_contexts(recorded);
      EXPECT_EQ(get_context(calls.captures.back(), &owner, &capture_green), CUDA_SUCCESS);
    };
    for (int i = 0; i < 5; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, green.stream));
      EXPECT_EQ(graph.is_captured(), i >= 1);
    }
    ASSERT_EQ(calls.captures.size(), 1u);
    EXPECT_EQ(capture_green, green.context) << "the graph for the green stream was recorded outside its context";
    ASSERT_FALSE(contexts.empty());
    EXPECT_EQ(contexts, std::vector<CUcontext>(contexts.size(), green_context)) << "a kernel would run outside it";
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(4, green.stream));
    EXPECT_EQ(calls.malloc_calls, 0u);
  }
  CudaCalls back;
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
    EXPECT_TRUE(graph.is_captured());
  }
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 3, green.stream));
  EXPECT_TRUE(back.captures.empty());
  EXPECT_EQ(back.launches, std::vector<cudaStream_t>({streams_[0], streams_[0], streams_[0], green.stream}));
}
// A green context stream on another device than the engine's takes the ordinary path, which refuses it
// as it refuses any stream there, and the engine keeps replaying on its own device.
TEST_F(ExecutionGraphReplayTest, GreenContextStreamOnAnotherDeviceKeepsTheGraph) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices for a green context on another device";
  }
  if (!green_contexts_supported(1)) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  GreenStream other;
  ASSERT_NO_FATAL_FAILURE(other.create(1));
  {
    CudaCalls calls;
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, other.stream), Error::InvalidState);
    }
    EXPECT_TRUE(calls.captures.empty());
  }
  EXPECT_TRUE(graph.is_captured());
  CudaCalls back;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 5, streams_[0]));
  EXPECT_TRUE(back.captures.empty());
  EXPECT_EQ(back.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

// A plain stream while a green context is current takes the ordinary path, since a capture stream made in
// the current context would be green, and the engine keeps the graph it has.
TEST_F(ExecutionGraphReplayTest, PlainStreamWithAGreenContextCurrentKeepsTheGraph) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green;
  ASSERT_NO_FATAL_FAILURE(green.create());
  const auto from_green = driver_entry<PFN_cuCtxFromGreenCtx_v12040>("cuCtxFromGreenCtx", 12040);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(from_green, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  CUcontext primary = nullptr;
  CUcontext green_context = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(from_green(&green_context, green.context), CUDA_SUCCESS);
  {
    // Puts the primary context back even when an assertion below returns early.
    std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) { set_current(primary); });
    CudaCalls calls;
    ASSERT_EQ(set_current(green_context), CUDA_SUCCESS);
    for (int i = 0; i < 3; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 2, streams_[0]));
    }
    EXPECT_TRUE(calls.captures.empty());
    EXPECT_TRUE(calls.launches.empty());
  }
  CudaCalls back;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 5, streams_[0]));
  EXPECT_TRUE(back.captures.empty());
  EXPECT_EQ(back.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

// TensorRT cannot run an engine on a stream of another regular context, but a graph recorded in the
// engine's own context replays there, and other buffers make the engine record again in its own context.
TEST_F(ExecutionGraphReplayTest, StreamOfAnotherRegularContextReplaysTheGraph) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  // Puts the primary context back and destroys the other one even when an assertion returns early.
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    destroy_context(other);
  });
  cudaStream_t other_stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  {
    CudaCalls calls;
    for (int i = 0; i < 2; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
    }
    EXPECT_EQ(calls.captures.size(), 1u);
    ASSERT_TRUE(graph.is_captured());
  }
  const size_t count = 16;
  std::vector<float> input(count), actual(count);
  for (size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(i);
  }
  ASSERT_EQ(cudaMemcpy(inputs_[0], input.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(set_current(other), CUDA_SUCCESS);
  {
    CudaCalls calls;
    for (int i = 0; i < 3; ++i) {
      ASSERT_EQ(cudaMemsetAsync(outputs_[0], 0xff, count * sizeof(float), other_stream), cudaSuccess);
      ASSERT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, other_stream), Error::Ok);
      ASSERT_EQ(
          cudaMemcpyAsync(actual.data(), outputs_[0], count * sizeof(float), cudaMemcpyDeviceToHost, other_stream),
          cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
      for (size_t j = 0; j < count; ++j) {
        ASSERT_FLOAT_EQ(actual[j], input[j] * 2.0f + 1.0f) << "element " << j;
      }
    }
    EXPECT_TRUE(calls.captures.empty());
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(3, other_stream));
  }
  // Other buffers record again in the engine's own context, on buffers of the graph's own, and replay
  // here, as the release did. The engine's own context keeps replaying, through those buffers.
  ASSERT_EQ(cudaMemcpy(inputs_[1], input.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  {
    CudaCalls calls;
    for (int i = 0; i < 4; ++i) {
      ASSERT_EQ(cudaMemsetAsync(outputs_[1], 0xff, count * sizeof(float), other_stream), cudaSuccess);
      ASSERT_EQ(graph.run(inputs_[1], outputs_[1], 2, 8, other_stream), Error::Ok);
      ASSERT_EQ(
          cudaMemcpyAsync(actual.data(), outputs_[1], count * sizeof(float), cudaMemcpyDeviceToHost, other_stream),
          cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
      for (size_t j = 0; j < count; ++j) {
        ASSERT_FLOAT_EQ(actual[j], input[j] * 2.0f + 1.0f) << "element " << j;
      }
    }
    EXPECT_EQ(calls.captures.size(), 1u);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(4, other_stream));
  }
  // A caller in the other context usually allocates its buffers there. They are copied in and out of
  // the graph's buffers, which belong to the engine's context, on the caller's stream.
  void* owned_input = nullptr;
  void* owned_output = nullptr;
  ASSERT_EQ(cudaMalloc(&owned_input, count * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&owned_output, count * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(owned_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  {
    CudaCalls calls;
    ASSERT_EQ(graph.run(owned_input, owned_output, 2, 8, other_stream), Error::Ok);
    ASSERT_EQ(
        cudaMemcpyAsync(actual.data(), owned_output, count * sizeof(float), cudaMemcpyDeviceToHost, other_stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
    for (size_t j = 0; j < count; ++j) {
      ASSERT_FLOAT_EQ(actual[j], input[j] * 2.0f + 1.0f) << "element " << j;
    }
    EXPECT_TRUE(calls.captures.empty());
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, other_stream));
  }
  EXPECT_EQ(cudaFree(owned_input), cudaSuccess);
  EXPECT_EQ(cudaFree(owned_output), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
  // The graph and the buffers it switched to belong to the engine's context, so they outlive the other one.
  restore.reset();
  CudaCalls back;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 5, streams_[0]));
  EXPECT_TRUE(back.captures.empty());
  EXPECT_EQ(back.launches, std::vector<cudaStream_t>(1, streams_[0]));
  // Until its shapes or its profile change, each of those calls copies its input in and its output out.
  CudaCalls copied;
  ASSERT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, streams_[0]), Error::Ok);
  EXPECT_EQ(copied.memcpy_calls, 2u);
  EXPECT_EQ(copied.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

// A call from another regular context that cannot record fails, and the engine's own context keeps its
// graph and keeps replaying it.
TEST_F(ExecutionGraphReplayTest, FailedRecordingsFromAnotherRegularContextLeaveTheOwnGraph) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    destroy_context(other);
  });
  cudaStream_t other_stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  ASSERT_EQ(set_current(other), CUDA_SUCCESS);
  {
    CudaCalls calls;
    calls.fail_captures = 3;
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(graph.run(inputs_[1], outputs_[1], 2, 8, other_stream), Error::InvalidState);
    }
    EXPECT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
  }
  EXPECT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
  restore.reset();
  CudaCalls back;
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 2, streams_[0]));
  }
  EXPECT_TRUE(back.captures.empty());
  EXPECT_EQ(back.launches, std::vector<cudaStream_t>(3, streams_[0]));
}

// The delegate keeps the buffer it copies an unaligned input through for later calls, so it allocates it
// in the engine's context even when a call from another regular context grows it. The engine's own
// calls then keep working after that other context is destroyed.
TEST_F(ExecutionGraphReplayTest, ABufferGrownByAnotherContextsCallOutlivesThatContext) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  const auto get_attribute = driver_entry<PFN_cuPointerGetAttribute_v4000>("cuPointerGetAttribute", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  ASSERT_NE(get_attribute, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  bool other_destroyed = false;
  // Puts the primary context back and destroys the other one even when an assertion returns early.
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    if (!other_destroyed) {
      destroy_context(other);
    }
  });
  cudaStream_t other_stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  ASSERT_EQ(graph.handle()->cached_input_ptrs[0], nullptr);

  // One float past a 256-byte boundary, so the delegate copies the input through a buffer of its own.
  const size_t count = 16;
  float* const unaligned = static_cast<float*>(inputs_[1]) + 1;
  std::vector<float> input(count), actual(count);
  for (size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(i);
  }
  ASSERT_EQ(cudaMemcpy(unaligned, input.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(set_current(other), CUDA_SUCCESS);
  ASSERT_EQ(graph.run(unaligned, outputs_[1], 2, 8, other_stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
  void* const kept = graph.handle()->cached_input_ptrs[0];
  ASSERT_NE(kept, nullptr);
  CUcontext owner = nullptr;
  ASSERT_EQ(get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, reinterpret_cast<CUdeviceptr>(kept)), CUDA_SUCCESS);
  EXPECT_EQ(owner, primary) << "the kept input buffer belongs to the context of the call that grew it";
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
  if (owner != primary) {
    return;
  }
  ASSERT_EQ(destroy_context(other), CUDA_SUCCESS);
  other_destroyed = true;

  for (int i = 0; i < 2; ++i) {
    for (size_t j = 0; j < count; ++j) {
      input[j] = static_cast<float>(j + 100 * i);
    }
    ASSERT_EQ(cudaMemcpy(unaligned, input.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(outputs_[1], 0xff, count * sizeof(float)), cudaSuccess);
    ASSERT_EQ(graph.run(unaligned, outputs_[1], 2, 8, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    ASSERT_EQ(graph.handle()->cached_input_ptrs[0], kept);
    ASSERT_EQ(cudaMemcpy(actual.data(), outputs_[1], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t j = 0; j < count; ++j) {
      ASSERT_FLOAT_EQ(actual[j], input[j] * 2.0f + 1.0f) << "call " << i << " element " << j;
    }
  }
}

// The shared scratch's completion event belongs to the engine's context, so a pooled engine refuses a
// stream of another regular context before it runs, whether or not another thread holds the capture
// lock, and the engine's own calls keep replaying their graph with no copies.
TEST_F(ExecutionGraphReplayTest, PooledCallFromAnotherRegularContextFailsAndKeepsTheOwnGraph) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    destroy_context(other);
  });
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

  const SizesType rows = 8;
  const SizesType cols = 32;
  const auto blob = build_scratch_blob(rows, cols);
  ASSERT_FALSE(blob.empty());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  for (bool hold_lock : {false, true}) {
    SCOPED_TRACE(hold_lock ? "capture lock held" : "capture lock free");
    ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
    {
      LoadedGraphEngine engine;
      ASSERT_EQ(engine.load(blob), Error::Ok);
      ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
      const size_t count = static_cast<size_t>(rows) * cols;
      const std::vector<float> ones(count, 1.0f);
      std::vector<float> actual(count);
      for (int slot = 0; slot < 2; ++slot) {
        ASSERT_EQ(cudaMemcpy(inputs_[slot], ones.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
      }
      for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(engine.run(inputs_[0], outputs_[0], rows, cols, streams_[0]), Error::Ok);
      }
      ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
      ASSERT_TRUE(engine.is_captured());

      // Before the holder starts, so a failed assertion cannot return while the thread is joinable.
      ASSERT_EQ(set_current(other), CUDA_SUCCESS);
      cudaStream_t other_stream = nullptr;
      ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
      // Holds the capture lock the way a load does while the call below runs.
      std::promise<void> held;
      std::promise<void> done;
      std::thread holder;
      if (hold_lock) {
        holder = std::thread([&] {
          const std::shared_lock<std::shared_mutex> lock(cuda_graph_capture_mutex());
          held.set_value();
          done.get_future().wait();
        });
        held.get_future().wait();
      }
      Error error = Error::Ok;
      {
        CudaCalls calls;
        error = engine.run(inputs_[1], outputs_[1], rows, cols, other_stream);
        EXPECT_TRUE(calls.captures.empty());
        EXPECT_TRUE(calls.launches.empty());
      }
      EXPECT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
      EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
      if (hold_lock) {
        done.set_value();
        holder.join();
      }
      EXPECT_EQ(error, Error::InvalidState);
      ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

      {
        CudaCalls back;
        ASSERT_EQ(engine.run(inputs_[0], outputs_[0], rows, cols, streams_[0]), Error::Ok);
        EXPECT_TRUE(back.captures.empty());
        EXPECT_EQ(back.launches, std::vector<cudaStream_t>(1, streams_[0]));
        EXPECT_EQ(back.memcpy_calls, 0u);
      }
      ASSERT_EQ(cudaMemcpy(actual.data(), outputs_[0], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      for (float value : actual) {
        ASSERT_FLOAT_EQ(value, 1.0f / rows);
      }
    }
    EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
  }
}

// The refusal comes before the call claims the shared scratch, so a pool that would have to grow for
// it stays as it was, in the context that made it. That holds with replay off, before the engine has run
// in its own context, and from a thread with no current context, as it does for a replaying engine.
TEST_F(ExecutionGraphReplayTest, PooledCallFromAnotherRegularContextLeavesThePoolAlone) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    destroy_context(other);
  });
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

  const SizesType rows = 8;
  const SizesType cols = 32;
  const auto blob = build_scratch_blob(rows, cols);
  ASSERT_FALSE(blob.empty());
  struct Case {
    const char* name;
    bool replay;
    bool ran_in_own_context;
    bool from_thread_without_context;
  };
  const Case cases[] = {
      {"replay on", true, true, false},
      {"replay off", false, true, false},
      {"before the engine ran", true, false, false},
      {"thread with no current context", true, true, true},
  };
  for (const Case& c : cases) {
    SCOPED_TRACE(c.name);
    ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
    BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, c.replay)};
    ASSERT_EQ(set_options({options, 2}), Error::Ok);
    {
      LoadedGraphEngine engine;
      ASSERT_EQ(engine.load(blob), Error::Ok);
      ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
      ASSERT_EQ(engine.handle()->execution_graph != nullptr, c.replay);
      const int engine_device = engine.handle()->device_id;
      const size_t count = static_cast<size_t>(rows) * cols;
      const std::vector<float> ones(count, 1.0f);
      std::vector<float> actual(count);
      ASSERT_EQ(cudaMemcpy(inputs_[0], ones.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
      if (c.ran_in_own_context) {
        for (int i = 0; i < 2; ++i) {
          ASSERT_EQ(engine.run(inputs_[0], outputs_[0], rows, cols, streams_[0]), Error::Ok);
        }
        ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
      }
      // An empty pool would have to grow for the next call.
      ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
      ASSERT_EQ(shared_scratch_capacity_for_testing(engine_device), 0u);

      ASSERT_EQ(set_current(other), CUDA_SUCCESS);
      cudaStream_t other_stream = nullptr;
      ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
      Error error = Error::Ok;
      if (c.from_thread_without_context) {
        std::thread([&] { error = engine.run(inputs_[0], outputs_[1], rows, cols, other_stream); }).join();
      } else {
        error = engine.run(inputs_[0], outputs_[1], rows, cols, other_stream);
      }
      EXPECT_EQ(error, Error::InvalidState);
      EXPECT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
      EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
      ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
      EXPECT_EQ(shared_scratch_capacity_for_testing(engine_device), 0u);

      // The engine's own context grows the pool itself and keeps working.
      ASSERT_EQ(engine.run(inputs_[0], outputs_[0], rows, cols, streams_[0]), Error::Ok);
      ASSERT_EQ(cudaMemcpy(actual.data(), outputs_[0], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      for (float value : actual) {
        ASSERT_FLOAT_EQ(value, 1.0f / rows);
      }
      EXPECT_GT(shared_scratch_capacity_for_testing(engine_device), 0u);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The refusal leaves the CUDA error state alone. An error the caller left pending is not sticky, so
// clearing it would hide it from the caller for good.
TEST_F(ExecutionGraphReplayTest, ARefusedCallFromAnotherContextLeavesTheCallersPendingCudaErrorAlone) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    destroy_context(other);
  });
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

  const SizesType rows = 8;
  const SizesType cols = 32;
  const auto blob = build_scratch_blob(rows, cols);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);

    ASSERT_EQ(set_current(other), CUDA_SUCCESS);
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
    void* impossible = nullptr;
    ASSERT_EQ(cudaMalloc(&impossible, static_cast<std::size_t>(1) << 62), cudaErrorMemoryAllocation);
    const Error refused = engine.run(inputs_[0], outputs_[1], rows, cols, other_stream);
    // Read before any assertion can return with it still pending.
    const cudaError_t pending = cudaGetLastError();
    EXPECT_EQ(refused, Error::InvalidState);
    EXPECT_EQ(pending, cudaErrorMemoryAllocation)
        << "the refusal cleared an error this backend did not cause: got " << cudaGetErrorString(pending);
    EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The shared scratch is one per device, and its event lives in the context of the device's first pooled
// call. A pooled engine loaded in another regular context is refused before it runs, so it cannot grow
// the pool in a context that may be destroyed first.
TEST_F(ExecutionGraphReplayTest, APooledEngineLoadedInAnotherRegularContextIsRefused) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    destroy_context(other);
  });
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

  const auto small_blob = build_scratch_blob(8, 32);
  const auto large_blob = build_scratch_blob(64, 256);
  ASSERT_FALSE(small_blob.empty());
  ASSERT_FALSE(large_blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  const auto run = [](LoadedGraphEngine& engine, SizesType rows, SizesType cols, cudaStream_t stream) {
    std::vector<float> input(static_cast<size_t>(rows) * cols, 1.0f), output(input.size(), -1.0f);
    const Error error = engine.run(input.data(), output.data(), rows, cols, stream);
    if (error == Error::Ok) {
      for (float value : output) {
        EXPECT_FLOAT_EQ(value, 1.0f / rows);
      }
    }
    return error;
  };
  {
    LoadedGraphEngine own, elsewhere;
    ASSERT_EQ(own.load(small_blob), Error::Ok);
    ASSERT_EQ(run(own, 8, 32, streams_[0]), Error::Ok);
    const int engine_device = own.handle()->device_id;
    const size_t capacity = shared_scratch_capacity_for_testing(engine_device);
    ASSERT_GT(capacity, 0u);

    ASSERT_EQ(set_current(other), CUDA_SUCCESS);
    ASSERT_EQ(elsewhere.load(large_blob), Error::Ok);
    ASSERT_TRUE(elsewhere.handle()->claims_pooled_scratch);
    ASSERT_EQ(elsewhere.handle()->context, other);
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
    EXPECT_EQ(run(elsewhere, 64, 256, other_stream), Error::InvalidState);
    EXPECT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
    ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
    EXPECT_EQ(shared_scratch_capacity_for_testing(engine_device), capacity) << "the refused engine grew the pool";
    EXPECT_EQ(run(own, 8, 32, streams_[0]), Error::Ok);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The driver can give a new context the handle of a destroyed one, so the pool tells contexts apart by id. A
// pooled engine whose context has the handle of the pool's context but not its id is refused before it runs.
TEST_F(ExecutionGraphReplayTest, APooledEngineWhoseContextOnlySharesThePoolContextsHandleIsRefused) {
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  const auto run = [this](LoadedGraphEngine& engine) {
    std::vector<float> input(8 * 32, 1.0f), output(input.size(), -1.0f);
    const Error error = engine.run(input.data(), output.data(), 8, 32, streams_[0]);
    if (error == Error::Ok) {
      for (float value : output) {
        EXPECT_FLOAT_EQ(value, 1.0f / 8);
      }
    }
    return error;
  };
  {
    LoadedGraphEngine own;
    ASSERT_EQ(own.load(blob), Error::Ok);
    ASSERT_EQ(run(own), Error::Ok);
    const int engine_device = own.handle()->device_id;
    const size_t capacity = shared_scratch_capacity_for_testing(engine_device);
    {
      RenewedContextIds renewed_ids;
      LoadedGraphEngine renewed;
      ASSERT_EQ(renewed.load(blob), Error::Ok);
      ASSERT_EQ(renewed.handle()->context, own.handle()->context);
      ASSERT_NE(renewed.handle()->context_id, own.handle()->context_id);
      EXPECT_EQ(run(renewed), Error::InvalidState);
    }
    EXPECT_EQ(shared_scratch_capacity_for_testing(engine_device), capacity);
    EXPECT_EQ(run(own), Error::Ok);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The driver can give a destroyed context's handle to a new one. A pooled call on a stream whose context has the
// engine's context handle but not its id is refused before it sets its shapes or claims the shared scratch.
TEST_F(ExecutionGraphReplayTest, APooledCallOnAContextThatOnlySharesTheEnginesHandleIsRefused) {
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  std::vector<float> input(8 * 32, 1.0f), output(input.size(), -1.0f);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_NE(engine.handle()->context_id, 0u);
    const int device = engine.handle()->device_id;
    {
      RenewedContextIds renewed_ids;
      EXPECT_EQ(engine.run(input.data(), output.data(), 8, 32, streams_[0]), Error::InvalidState);
    }
    EXPECT_EQ(shared_scratch_capacity_for_testing(device), 0u) << "the refused call claimed the shared scratch";
    ASSERT_EQ(engine.run(input.data(), output.data(), 8, 32, streams_[0]), Error::Ok);
    for (float value : output) {
      EXPECT_FLOAT_EQ(value, 1.0f / 8);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// A first pooled call that cannot make the pool's event leaves nothing of the pool in its context, so the pool
// is not tied to that context: once it is destroyed, a pooled engine of another context still runs.
TEST_F(ExecutionGraphReplayTest, AFirstClaimThatMakesNoEventLeavesThePoolToAnotherContext) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    if (other != nullptr) {
      destroy_context(other);
    }
  });
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  // Earlier cases tied the pool to this process's primary context, and the reset leaves that, so the pool is
  // put back to a process that has made no pooled call yet, and back again at the end.
  SharedScratchDevice& pool = scratch_pool().get(device);
  unsigned long long kept_owner = 0;
  {
    std::lock_guard<std::mutex> lock(pool.mu);
    kept_owner = pool.context_id;
    pool.context_id = 0;
  }
  std::unique_ptr<void, std::function<void(void*)>> restore_owner(&kept_owner, [&](void*) {
    std::lock_guard<std::mutex> lock(pool.mu);
    pool.context_id = kept_owner;
  });
  const auto run = [](LoadedGraphEngine& engine, cudaStream_t stream) {
    std::vector<float> input(8 * 32, 1.0f), output(input.size(), -1.0f);
    const Error error = engine.run(input.data(), output.data(), 8, 32, stream);
    if (error == Error::Ok) {
      for (float value : output) {
        EXPECT_FLOAT_EQ(value, 1.0f / 8);
      }
    }
    return error;
  };

  ASSERT_EQ(set_current(other), CUDA_SUCCESS);
  {
    LoadedGraphEngine elsewhere;
    ASSERT_EQ(elsewhere.load(blob), Error::Ok);
    ASSERT_TRUE(elsewhere.handle()->claims_pooled_scratch);
    ASSERT_EQ(elsewhere.handle()->context, other);
    cudaStream_t other_stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&other_stream), cudaSuccess);
    {
      CudaCalls calls;
      calls.fail_event_creates = 1;
      EXPECT_EQ(run(elsewhere, other_stream), Error::Internal);
      EXPECT_EQ(calls.fail_event_creates, 0u) << "the call made no event to fail";
    }
    EXPECT_EQ(cudaStreamSynchronize(other_stream), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(other_stream), cudaSuccess);
  }
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
  ASSERT_EQ(destroy_context(other), CUDA_SUCCESS);
  other = nullptr;

  {
    LoadedGraphEngine own;
    ASSERT_EQ(own.load(blob), Error::Ok);
    EXPECT_EQ(run(own, streams_[0]), Error::Ok)
        << "the pool stayed tied to a context its failed first claim left empty";
    std::lock_guard<std::mutex> lock(pool.mu);
    EXPECT_EQ(pool.context_id, own.handle()->context_id);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// An engine loaded while a green context is current is created in the device's primary context, so its
// pooled calls run, a buffer it grows after the green context is destroyed lives in a context that is still
// there, and TensorRT can destroy the engine after the green context without an error.
TEST_F(ExecutionGraphReplayTest, AnEngineLoadedUnderAGreenContextOutlivesIt) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  const auto from_green = driver_entry<PFN_cuCtxFromGreenCtx_v12040>("cuCtxFromGreenCtx", 12040);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  ASSERT_NE(from_green, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  CUcontext primary = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  auto green = std::make_unique<GreenStream>();
  ASSERT_NO_FATAL_FAILURE(green->create());
  CUcontext green_context = nullptr;
  ASSERT_EQ(from_green(&green_context, green->context), CUDA_SUCCESS);
  // Once per process, since a second registration logs an error.
  static const bool counting = register_pal(PalImpl::create(count_tensorrt_errors, __FILE__));
  ASSERT_TRUE(counting);
  const int errors = tensorrt_errors.load();
  const int retains = primary_context_retains.load();
  const int releases = primary_context_releases.load();

  const SizesType rows = 2;
  const SizesType cols = 8;
  const auto blob = build_scratch_blob(rows, cols);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(set_current(green_context), CUDA_SUCCESS);
    const Error loaded = engine.load(blob);
    CUcontext after_load = nullptr;
    const CUresult read = get_current(&after_load);
    ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
    ASSERT_EQ(loaded, Error::Ok);
    ASSERT_EQ(read, CUDA_SUCCESS);
    EXPECT_EQ(after_load, green_context) << "the load did not give the caller its green context back";
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
    EXPECT_EQ(engine.handle()->context, primary);

    const size_t count = static_cast<size_t>(rows) * cols;
    const std::vector<float> ones(count, 1.0f);
    std::vector<float> actual(count);
    const auto check = [&](void* input, cudaStream_t stream) {
      ASSERT_EQ(cudaMemcpy(input, ones.data(), count * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemset(outputs_[0], 0, count * sizeof(float)), cudaSuccess);
      ASSERT_EQ(engine.run(input, outputs_[0], rows, cols, stream), Error::Ok);
      ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(actual.data(), outputs_[0], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      for (float value : actual) {
        ASSERT_FLOAT_EQ(value, 1.0f / rows);
      }
    };
    ASSERT_NO_FATAL_FAILURE(check(inputs_[0], green->stream));
    ASSERT_NO_FATAL_FAILURE(check(inputs_[0], streams_[0]));
    green.reset();
    // One float past a 256-byte boundary, so the delegate grows a buffer of its own to copy the input through.
    ASSERT_NO_FATAL_FAILURE(check(static_cast<float*>(inputs_[1]) + 1, streams_[0]));
  }
  EXPECT_EQ(tensorrt_errors.load(), errors);
  EXPECT_EQ(primary_context_retains.load() - retains, 1);
  EXPECT_EQ(primary_context_releases.load() - releases, 1) << "the engine kept the primary context it was created in";
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The pool keeps its disposal stream for every later growth, so a growth while a green context is current
// makes it in the engine's context: a stream made in the green context would go away with it.
TEST_F(ExecutionGraphReplayTest, APoolGrowthUnderAGreenContextMakesItsDisposalStreamInTheEnginesContext) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  const auto from_green = driver_entry<PFN_cuCtxFromGreenCtx_v12040>("cuCtxFromGreenCtx", 12040);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  const auto get_stream_context = driver_entry<PFN_cuStreamGetCtx_v12050>("cuStreamGetCtx", 12050);
  ASSERT_NE(from_green, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  ASSERT_NE(get_stream_context, nullptr);
  CUcontext primary = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  GreenStream green;
  ASSERT_NO_FATAL_FAILURE(green.create());
  CUcontext green_context = nullptr;
  ASSERT_EQ(from_green(&green_context, green.context), CUDA_SUCCESS);

  const auto small_blob = build_scratch_blob(8, 32);
  const auto large_blob = build_scratch_blob(64, 256);
  ASSERT_FALSE(small_blob.empty());
  ASSERT_FALSE(large_blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine smaller, larger;
    ASSERT_EQ(smaller.load(small_blob), Error::Ok);
    ASSERT_EQ(larger.load(large_blob), Error::Ok);
    ASSERT_TRUE(smaller.handle()->claims_pooled_scratch);
    ASSERT_TRUE(larger.handle()->claims_pooled_scratch);
    std::vector<float> small_in(8 * 32, 1.0f), small_out(small_in.size(), -1.0f);
    std::vector<float> large_in(64 * 256, 1.0f), large_out(large_in.size(), -1.0f);
    ASSERT_EQ(set_current(green_context), CUDA_SUCCESS);
    const Error first = smaller.run(small_in.data(), small_out.data(), 8, 32, green.stream);
    const Error second = larger.run(large_in.data(), large_out.data(), 64, 256, green.stream);
    ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
    ASSERT_EQ(first, Error::Ok);
    ASSERT_EQ(second, Error::Ok);
    for (float value : large_out) {
      ASSERT_FLOAT_EQ(value, 1.0f / 64);
    }
    cudaStream_t disposal_stream = nullptr;
    {
      SharedScratchDevice& pool = scratch_pool().get(larger.handle()->device_id);
      std::lock_guard<std::mutex> lock(pool.mu);
      disposal_stream = pool.disposal_stream;
    }
    ASSERT_NE(disposal_stream, nullptr) << "the larger engine did not grow the pool";
    CUcontext owner = nullptr;
    CUgreenCtx green_owner = nullptr;
    ASSERT_EQ(get_stream_context(disposal_stream, &owner, &green_owner), CUDA_SUCCESS);
    EXPECT_EQ(green_owner, nullptr) << "the disposal stream belongs to the green context";
    EXPECT_EQ(owner, primary);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The pool's event and buffer are made in the engine's context, so a pooled call from a thread where
// another regular context is current leaves nothing of the pool in that context.
TEST_F(ExecutionGraphReplayTest, APooledCallWithAnotherContextCurrentMakesThePoolInTheEnginesContext) {
  const auto create_context = driver_entry<PFN_cuCtxCreate_v3020>("cuCtxCreate", 3020);
  const auto destroy_context = driver_entry<PFN_cuCtxDestroy_v4000>("cuCtxDestroy", 4000);
  const auto get_current = driver_entry<PFN_cuCtxGetCurrent_v4000>("cuCtxGetCurrent", 4000);
  const auto set_current = driver_entry<PFN_cuCtxSetCurrent_v4000>("cuCtxSetCurrent", 4000);
  const auto get_attribute = driver_entry<PFN_cuPointerGetAttribute_v4000>("cuPointerGetAttribute", 4000);
  ASSERT_NE(create_context, nullptr);
  ASSERT_NE(destroy_context, nullptr);
  ASSERT_NE(get_current, nullptr);
  ASSERT_NE(set_current, nullptr);
  ASSERT_NE(get_attribute, nullptr);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  CUcontext primary = nullptr;
  CUcontext other = nullptr;
  ASSERT_EQ(get_current(&primary), CUDA_SUCCESS);
  ASSERT_EQ(create_context(&other, 0, device), CUDA_SUCCESS);
  bool other_destroyed = false;
  std::unique_ptr<void, std::function<void(void*)>> restore(&primary, [&](void*) {
    set_current(primary);
    if (!other_destroyed) {
      destroy_context(other);
    }
  });
  ASSERT_EQ(set_current(primary), CUDA_SUCCESS);

  const SizesType rows = 8;
  const SizesType cols = 32;
  const auto blob = build_scratch_blob(rows, cols);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  const auto run = [&](LoadedGraphEngine& engine) {
    std::vector<float> input(static_cast<size_t>(rows) * cols, 1.0f), output(input.size(), -1.0f);
    const Error error = engine.run(input.data(), output.data(), rows, cols, streams_[0]);
    if (error == Error::Ok) {
      for (float value : output) {
        EXPECT_FLOAT_EQ(value, 1.0f / rows);
      }
    }
    return error;
  };
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
    SharedScratchDevice& pool = scratch_pool().get(engine.handle()->device_id);
    {
      // The flag outlives a reset, so it is dropped here (and leaked) to make this call create one.
      std::lock_guard<std::mutex> lock(pool.mu);
      pool.marker.flag = nullptr;
      pool.marker.sequence = 0;
    }
    ASSERT_EQ(set_current(other), CUDA_SUCCESS);
    const Error first = run(engine);
    ASSERT_EQ(set_current(primary), CUDA_SUCCESS);
    ASSERT_EQ(first, Error::Ok);
    void* buffer = nullptr;
    void* flag = nullptr;
    {
      std::lock_guard<std::mutex> lock(pool.mu);
      buffer = pool.buffer;
      flag = pool.marker.flag;
    }
    ASSERT_NE(buffer, nullptr);
    CUcontext owner = nullptr;
    ASSERT_EQ(get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, reinterpret_cast<CUdeviceptr>(buffer)), CUDA_SUCCESS);
    EXPECT_EQ(owner, primary) << "the pool's buffer belongs to the context that was current at the call";
    if (device_can_wait_on_a_flag()) {
      ASSERT_NE(flag, nullptr);
      ASSERT_EQ(get_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, reinterpret_cast<CUdeviceptr>(flag)), CUDA_SUCCESS);
      EXPECT_EQ(owner, primary) << "the pool's flag belongs to the context that was current at the call";
    }
    ASSERT_EQ(destroy_context(other), CUDA_SUCCESS);
    other_destroyed = true;
    EXPECT_EQ(run(engine), Error::Ok);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The driver often gives a new green context the handle of one just destroyed. The old graph lived in the
// destroyed context, so the new one must record its own.
TEST_F(ExecutionGraphReplayTest, AGreenContextOnADestroyedOnesHandleRecordsAgain) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  // Declared before the engines, so the green context the graph is recorded in outlives them, as TensorRT
  // requires.
  GreenStream green;
  ASSERT_NO_FATAL_FAILURE(green.create());
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  int seed = 0;
  // Only the warm-up call, as a call on the destroyed context would leave.
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed++, green.stream));
  // The same handle with a new id is what a new green context given the destroyed one's handle looks like.
  RenewedContextIds renewed_ids;
  CudaCalls calls;
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed++, green.stream));
  }
  EXPECT_EQ(calls.captures.size(), 1u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>({green.stream}));
}

// Beyond 4 green contexts, the one called longest ago gives up its graph to the new one.
TEST_F(ExecutionGraphReplayTest, AFifthGreenContextTakesThePlaceOfTheLeastRecentlyUsed) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green[5];
  for (GreenStream& each : green) {
    ASSERT_NO_FATAL_FAILURE(each.create());
  }
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  int seed = 0;
  // The first context is called again before the fifth arrives, so the second is the one called longest ago.
  for (int index : {0, 1, 2, 3, 0, 4}) {
    for (int i = 0; i < 2; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed++, green[index].stream));
    }
  }
  CudaCalls calls;
  for (int index : {0, 1, 1}) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed++, green[index].stream));
  }
  EXPECT_EQ(calls.captures.size(), 1u) << "the second green context should record again, the first replay";
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>({green[0].stream, green[1].stream}));
}

// A staged green graph copies through buffers of its own. The green context that takes its place frees them
// on its stream, not with a device-wide cudaFree.
TEST_F(ExecutionGraphReplayTest, AStagedGreenGraphFreesItsBuffersOnTheStreamWhenItsPlaceIsTaken) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green[5];
  for (GreenStream& each : green) {
    ASSERT_NO_FATAL_FAILURE(each.create());
  }
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(stage(plain, graph, 2, 8, green[0].stream));
  int seed = 10;
  for (int index : {1, 2, 3}) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed++, green[index].stream));
  }
  CudaCalls calls;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, seed++, green[4].stream));
  EXPECT_EQ(calls.free_calls, 0u);
  EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, green[4].stream));
}

// TensorRT may run some kernels of an engine on auxiliary streams. A recording holds TensorRT's join of
// those streams as a dependency inside the graph, so a pooled engine with them replays with the outputs of
// its plain calls, on a plain stream and on a green context's stream.
TEST_F(ExecutionGraphReplayTest, AnEngineWithAuxiliaryStreamsReplaysWithItsPlainOutputs) {
  const SizesType rows = 128;
  const SizesType cols = 256;
  const auto blob = build_aux_stream_blob(rows, cols);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  const bool green_supported = green_contexts_supported();
  {
    // Declared before the engines, which TensorRT requires to be destroyed first.
    GreenStream green;
    BackendOption plain_options[] = {option(kScratchOption, true), option(kGraphOption, false)};
    ASSERT_EQ(set_options({plain_options, 2}), Error::Ok);
    LoadedGraphEngine plain;
    ASSERT_EQ(plain.load(blob), Error::Ok);
    BackendOption graph_options[] = {option(kScratchOption, true), option(kGraphOption, true)};
    ASSERT_EQ(set_options({graph_options, 2}), Error::Ok);
    LoadedGraphEngine graph;
    ASSERT_EQ(graph.load(blob), Error::Ok);
    ASSERT_TRUE(graph.handle()->claims_pooled_scratch);
    if (graph.handle()->engine->getNbAuxStreams() == 0) {
      GTEST_SKIP() << "TensorRT built this engine without auxiliary streams";
    }
    const size_t count = static_cast<size_t>(rows) * cols;
    std::vector<float> input(count), expected(count), actual(count);
    std::vector<cudaStream_t> streams = {streams_[0]};
    if (green_supported) {
      ASSERT_NO_FATAL_FAILURE(green.create());
      streams.push_back(green.stream);
    }
    int seed = 0;
    for (cudaStream_t stream : streams) {
      SCOPED_TRACE(stream == streams_[0] ? "plain stream" : "green context stream");
      CudaCalls calls;
      for (int i = 0; i < 3; ++i, ++seed) {
        // A new input on every call, so a replay that does no work cannot pass with an earlier call's output.
        for (size_t j = 0; j < count; ++j) {
          input[j] = static_cast<float>(static_cast<int>((j * 13 + seed * 5) % 29) - 14) / 14.0f;
        }
        std::fill(expected.begin(), expected.end(), -1.0f);
        ASSERT_EQ(plain.run(input.data(), expected.data(), rows, cols, streams_[0]), Error::Ok);
        std::fill(actual.begin(), actual.end(), -1.0f);
        ASSERT_EQ(graph.run(input.data(), actual.data(), rows, cols, stream), Error::Ok);
        for (size_t j = 0; j < count; ++j) {
          ASSERT_NEAR(actual[j], expected[j], 1e-5f) << "call " << i << " element " << j;
        }
      }
      EXPECT_EQ(calls.captures.size(), 1u);
      EXPECT_EQ(calls.launches.size(), 2u);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
  if (!green_supported) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU, so only the plain stream was checked";
  }
}

// A graph does not need the stream it was recorded on, and one kept in a green context could not be destroyed
// once a caller destroyed that context.
TEST_F(ExecutionGraphReplayTest, EachRecordingDestroysItsCaptureStream) {
  const bool green_supported = green_contexts_supported();
  GreenStream green;
  std::vector<cudaStream_t> streams{streams_[0]};
  if (green_supported) {
    ASSERT_NO_FATAL_FAILURE(green.create());
    streams.push_back(green.stream);
  }
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  for (cudaStream_t stream : streams) {
    CudaCalls calls;
    for (int i = 0; i < 2; ++i) {
      ASSERT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, stream), Error::Ok);
    }
    ASSERT_TRUE(graph.is_captured());
    ASSERT_EQ(calls.captures.size(), 1u);
    EXPECT_EQ(calls.destroyed_streams, calls.captures);
  }
  if (!green_supported) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU, so only the plain stream was checked";
  }
}

// TensorRT requires a green context to outlive the graphs recorded in it, so the engine is destroyed first.
// That releases each green graph it kept, as does a newer green context taking the place of one.
TEST_F(ExecutionGraphReplayTest, AnEngineDestroyedBeforeItsGreenContextsReleasesTheirGraphs) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green[5];
  for (GreenStream& each : green) {
    ASSERT_NO_FATAL_FAILURE(each.create());
  }
  LoadedGraphEngine plain;
  auto graph = std::make_unique<LoadedGraphEngine>();
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, *graph));
  int seed = 0;
  {
    CudaCalls calls;
    // The fifth of these takes the place of the first one's graph.
    for (GreenStream& each : green) {
      for (int i = 0; i < 2; ++i) {
        ASSERT_NO_FATAL_FAILURE(compare(plain, *graph, 0, 2, 8, seed++, each.stream));
        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
      }
    }
    EXPECT_EQ(calls.graph_exec_destroys, std::vector<cudaError_t>({cudaSuccess}));
  }
  CudaCalls calls;
  graph.reset();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(calls.graph_exec_destroys, std::vector<cudaError_t>(4, cudaSuccess));
}

// A failed release is logged and cleared, so the caller's next CUDA call does not report it.
TEST_F(ExecutionGraphReplayTest, AFailedGraphReleaseLeavesNoErrorBehind) {
  LoadedGraphEngine plain;
  auto graph = std::make_unique<LoadedGraphEngine>();
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, *graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, *graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph->is_captured());
  CudaCalls calls;
  calls.fail_graph_exec_destroys = true;
  graph.reset();
  ASSERT_EQ(calls.graph_exec_destroys.size(), 1u);
  EXPECT_NE(calls.graph_exec_destroys[0], cudaSuccess);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// After 3 failed recordings a green context stops recording, and that says nothing about the caller's other
// context.
TEST_F(ExecutionGraphReplayTest, FailedGreenRecordingsLeaveThePlainStreamReplaying) {
  if (!green_contexts_supported()) {
    GTEST_SKIP() << "Green contexts are not supported by this driver or GPU";
  }
  GreenStream green;
  ASSERT_NO_FATAL_FAILURE(green.create());
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  {
    CudaCalls calls;
    calls.fail_captures = 3;
    // A warm-up, 3 failed recordings, then a plain call.
    for (int i = 0; i < 5; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 2, green.stream));
    }
    EXPECT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
  }
  CudaCalls back;
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, i + 7, streams_[0]));
  }
  EXPECT_TRUE(back.captures.empty());
  EXPECT_EQ(back.launches, std::vector<cudaStream_t>(3, streams_[0]));
}
// The pool frees its old buffer when a larger request grows it, so a graph recorded on that buffer
// must not replay.
TEST_F(ExecutionGraphReplayTest, PooledScratchRecordsAgainWhenThePoolGrows) {
  const auto small_blob = build_scratch_blob(8, 32);
  const auto large_blob = build_scratch_blob(64, 256);
  ASSERT_FALSE(small_blob.empty());
  ASSERT_FALSE(large_blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine smaller, larger;
    ASSERT_EQ(smaller.load(small_blob), Error::Ok);
    ASSERT_EQ(larger.load(large_blob), Error::Ok);
    ASSERT_TRUE(smaller.handle()->claims_pooled_scratch);
    ASSERT_TRUE(larger.handle()->claims_pooled_scratch);
    const int device = smaller.handle()->device_id;
    // The buffers stay put: a GPU that reads host memory directly binds them, so a new buffer on each call
    // would record again.
    std::vector<float> small_input(8 * 32, 1.0f), small_output(small_input.size());
    std::vector<float> large_input(64 * 256, 1.0f), large_output(large_input.size());
    const auto run = [&](LoadedGraphEngine& engine, SizesType rows, SizesType cols) {
      std::vector<float>& input = rows == 8 ? small_input : large_input;
      std::vector<float>& output = rows == 8 ? small_output : large_output;
      output.assign(output.size(), -1.0f);
      ASSERT_EQ(engine.run(input.data(), output.data(), rows, cols, streams_[0]), Error::Ok);
      for (float value : output) {
        ASSERT_FLOAT_EQ(value, 1.0f / rows);
      }
    };
    for (int i = 0; i < 3; ++i) {
      ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32));
    }
    ASSERT_TRUE(smaller.is_captured());
    const size_t capacity = shared_scratch_capacity_for_testing(device);
    ASSERT_NO_FATAL_FAILURE(run(larger, 64, 256));
    ASSERT_GT(shared_scratch_capacity_for_testing(device), capacity) << "the larger engine did not grow the pool";

    CudaCalls calls;
    ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32));
    EXPECT_FALSE(smaller.is_captured()) << "the graph recorded on the freed buffer was kept";
    EXPECT_TRUE(calls.launches.empty());
    for (int i = 0; i < 3; ++i) {
      ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32));
      ASSERT_NO_FATAL_FAILURE(run(larger, 64, 256));
    }
    EXPECT_TRUE(smaller.is_captured());
    EXPECT_TRUE(larger.is_captured());
    EXPECT_EQ(calls.captures.size(), 2u);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(6, streams_[0]));
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// A pool growth has to drop every graph recorded on the old buffer, whichever stream the next call uses,
// including graphs recorded in a green context.
TEST_F(ExecutionGraphReplayTest, PooledScratchGrowthDropsTheGraphOnOtherStreamShapes) {
  // Only the green context cases need a green context, so the first case runs everywhere.
  std::unique_ptr<GreenStream> green;
  if (green_contexts_supported()) {
    green = std::make_unique<GreenStream>();
    ASSERT_NO_FATAL_FAILURE(green->create());
  }
  const auto small_blob = build_scratch_blob(8, 32);
  const auto large_blob = build_scratch_blob(64, 256);
  ASSERT_FALSE(small_blob.empty());
  ASSERT_FALSE(large_blob.empty());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  const auto run = [](LoadedGraphEngine& engine, SizesType rows, SizesType cols, std::optional<cudaStream_t> stream) {
    std::vector<float> input(static_cast<size_t>(rows) * cols, 1.0f), output(input.size(), -1.0f);
    ASSERT_EQ(engine.run(input.data(), output.data(), rows, cols, stream), Error::Ok);
    for (float value : output) {
      ASSERT_FLOAT_EQ(value, 1.0f / rows);
    }
  };
  std::vector<std::pair<std::optional<cudaStream_t>, std::optional<cudaStream_t>>> cases = {
      {std::nullopt, std::nullopt}};
  if (green) {
    cases.push_back({green->stream, green->stream});
    cases.push_back({streams_[0], green->stream});
  }
  for (const auto& [record_on, after_growth_on] : cases) {
    SCOPED_TRACE(
        ::testing::Message() << "record on " << static_cast<const void*>(record_on.value_or(nullptr)) << ", then "
                             << static_cast<const void*>(after_growth_on.value_or(nullptr)));
    ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
    LoadedGraphEngine smaller, larger;
    ASSERT_EQ(smaller.load(small_blob), Error::Ok);
    ASSERT_EQ(larger.load(large_blob), Error::Ok);
    ASSERT_TRUE(smaller.handle()->claims_pooled_scratch);
    ASSERT_TRUE(larger.handle()->claims_pooled_scratch);
    for (int i = 0; i < 3; ++i) {
      ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32, record_on));
    }
    ASSERT_TRUE(smaller.is_captured());
    const int device = smaller.handle()->device_id;
    const size_t capacity = shared_scratch_capacity_for_testing(device);
    ASSERT_NO_FATAL_FAILURE(run(larger, 64, 256, streams_[1]));
    ASSERT_GT(shared_scratch_capacity_for_testing(device), capacity) << "the larger engine did not grow the pool";
    {
      CudaCalls calls;
      ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32, after_growth_on));
      if (after_growth_on == record_on) {
        EXPECT_FALSE(smaller.is_captured()) << "the graph recorded on the freed buffer was kept";
      }
      ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32, record_on));
      EXPECT_EQ(calls.launches.size(), calls.captures.size()) << "a graph recorded on the freed buffer was replayed";
    }
    for (int i = 0; i < 2; ++i) {
      ASSERT_NO_FATAL_FAILURE(run(smaller, 8, 32, record_on));
    }
    EXPECT_TRUE(smaller.is_captured());
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// A staged replay copies its outputs out after the graph launch, so when that copy fails the engine's
// work is still queued on the pooled buffer. The stream has to drain before the pool's lock is released.
TEST_F(ExecutionGraphReplayTest, AFailedPooledReplayDrainsBeforeReleasingThePool) {
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
    std::vector<float> input(kMaxElements, 1.0f), output(kMaxElements, -1.0f);
    for (void* device_input : inputs_) {
      ASSERT_EQ(cudaMemcpy(device_input, input.data(), kMaxBytes, cudaMemcpyHostToDevice), cudaSuccess);
    }
    // Moving between the two slots twice puts the replay on the graph's own buffers.
    for (int slot : {0, 0, 1, 0, 1}) {
      ASSERT_EQ(engine.run(inputs_[slot], outputs_[slot], 8, 32, streams_[0]), Error::Ok);
    }
    ASSERT_TRUE(engine.is_captured());
    ASSERT_NE(engine.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[1]);
    SharedScratchDevice& device = scratch_pool().get(engine.handle()->device_id);
    bool launched = false;
    std::optional<bool> held_at_drain;
    CudaCalls calls;
    calls.after_launch = [&](cudaStream_t) {
      launched = true;
      calls.fail_memcpy_call = calls.memcpy_calls + 1;
    };
    calls.before_synchronize = [&](cudaStream_t stream) {
      if (!launched || held_at_drain.has_value()) {
        return;
      }
      EXPECT_EQ(stream, streams_[0]);
      // try_lock on the thread that owns the lock is undefined, so the probe runs on another thread.
      held_at_drain = !std::async(std::launch::async, [&] {
                         const bool lock_free = device.mu.try_lock();
                         if (lock_free) {
                           device.mu.unlock();
                         }
                         return lock_free;
                       }).get();
    };
    EXPECT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::Internal);
    calls.after_launch = nullptr;
    calls.before_synchronize = nullptr;
    ASSERT_TRUE(launched);
    ASSERT_TRUE(held_at_drain.has_value()) << "the failed call never drained its stream";
    EXPECT_TRUE(*held_at_drain) << "the pool's lock was released with the failed call's work still queued";
    ASSERT_EQ(cudaMemset(outputs_[1], 0xff, kMaxBytes), cudaSuccess);
    ASSERT_EQ(engine.run(inputs_[1], outputs_[1], 8, 32, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaMemcpy(output.data(), outputs_[1], kMaxBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (float value : output) {
      ASSERT_FLOAT_EQ(value, 1.0f / 8);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// cudaMalloc does not clear memory, and the handoff's wait compares cyclically, so the pool has to
// clear its flag before any claimant can read it. Here the flag is made while the first claimant's
// stream is held, from memory filled with a pattern, and is read back once that call has released the
// pool. A clear queued on the held stream has not run yet, so the pattern is still there.
TEST_F(ExecutionGraphTest, TheHandoffFlagIsClearedBeforeAnyClaimantCanReadIt) {
  if (!device_can_wait_on_a_flag()) {
    GTEST_SKIP() << "This GPU has no 64-bit stream memory operations, so the pool has no flag";
  }
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  LoadedGraphEngine engine;
  ASSERT_EQ(engine.load(blob), Error::Ok);
  ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
  SharedScratchDevice& device = scratch_pool().get(engine.handle()->device_id);
  {
    // The flag outlives a reset, so it is dropped here (and leaked) to make the next claim create one.
    std::lock_guard<std::mutex> lock(device.mu);
    device.marker.flag = nullptr;
    device.marker.sequence = 0;
  }

  struct Gate {
    std::atomic<bool> open{false};
  } gate;
  ASSERT_EQ(
      cudaLaunchHostFunc(
          streams_[0],
          [](void* data) {
            // Bounded, so a case that fails before opening the gate cannot hang the stream.
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!static_cast<Gate*>(data)->open.load() && std::chrono::steady_clock::now() < give_up) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
          },
          &gate),
      cudaSuccess);
  Error result = Error::Internal;
  bool clear_waited_for = false;
  std::thread caller([&] {
    CudaCalls calls;
    calls.fill_allocations_with = 0x55;
    result = engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]);
    // The clear runs on a stream of its own, which is destroyed once the host has waited for it.
    for (const CudaCall& call : calls.operations) {
      clear_waited_for |= call.kind == CudaCallKind::Synchronize && call.stream != streams_[0] &&
          std::count(calls.destroyed_streams.begin(), calls.destroyed_streams.end(), call.stream) > 0;
    }
  });

  // The call holds the pool's lock from its claim until it has queued its write to the flag.
  void* flag = nullptr;
  unsigned long long sequence = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (flag == nullptr && std::chrono::steady_clock::now() < deadline) {
    if (device.mu.try_lock()) {
      if (device.marker.sequence > 0) {
        flag = device.marker.flag;
        sequence = device.marker.sequence;
      }
      device.mu.unlock();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  unsigned long long value = 0;
  cudaStream_t read_stream = nullptr;
  cudaError_t read_error = cudaErrorUnknown;
  if (flag != nullptr && cudaStreamCreateWithFlags(&read_stream, cudaStreamNonBlocking) == cudaSuccess) {
    read_error = cudaMemcpyAsync(&value, flag, sizeof(value), cudaMemcpyDeviceToHost, read_stream);
    if (read_error == cudaSuccess) {
      read_error = cudaStreamSynchronize(read_stream);
    }
    EXPECT_EQ(cudaStreamDestroy(read_stream), cudaSuccess);
  }
  gate.open.store(true);
  caller.join();

  ASSERT_NE(flag, nullptr) << "the call never recorded its enqueue on the pool's flag";
  ASSERT_EQ(read_error, cudaSuccess);
  EXPECT_TRUE(clear_waited_for) << "the host did not wait for the flag's clear, so a claimant could read it first";
  EXPECT_EQ(sequence, 1u);
  EXPECT_EQ(value, 0u) << "the flag still held what its allocation left there after the call that made it had "
                          "released the pool, so another claimant could have read that value";
  EXPECT_EQ(result, Error::Ok);
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// Puts a device's pool on the path a GPU without 64-bit stream memory operations takes, and puts it back.
class WithoutTheHandoffFlag {
 public:
  explicit WithoutTheHandoffFlag(SharedScratchDevice& device) : device_(device) {
    std::lock_guard<std::mutex> lock(device_.mu);
    kept_flag_ = device_.marker.flag;
    kept_event_wait_only_ = device_.marker.event_wait_only;
    device_.marker.flag = nullptr;
    device_.marker.event_wait_only = true;
  }
  ~WithoutTheHandoffFlag() {
    std::lock_guard<std::mutex> lock(device_.mu);
    device_.marker.flag = kept_flag_;
    device_.marker.event_wait_only = kept_event_wait_only_;
  }
  WithoutTheHandoffFlag(const WithoutTheHandoffFlag&) = delete;
  WithoutTheHandoffFlag& operator=(const WithoutTheHandoffFlag&) = delete;

 private:
  SharedScratchDevice& device_;
  void* kept_flag_ = nullptr;
  bool kept_event_wait_only_ = false;
};

// On a GPU without 64-bit stream memory operations the pool has no flag, so a dropped event wait could let the
// next claimant start on the scratch while the call before it still runs. Here the pool is put on that path, the
// earlier call is held on its stream, and the next pooled call, on another stream, has its event wait dropped:
// the earlier call keeps the pool's lock until it has finished, so the next call cannot return before it.
TEST_F(ExecutionGraphTest, WithoutAFlagAPooledCallHoldsThePoolUntilItHasFinished) {
  constexpr std::chrono::milliseconds kWatchTheSecondCallFor{500};
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  LoadedGraphEngine engines[2];
  std::vector<float> input(kMaxElements, 1.0f), output(kMaxElements);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(engines[i].load(blob), Error::Ok);
    ASSERT_TRUE(engines[i].handle()->claims_pooled_scratch);
    ASSERT_EQ(cudaMemcpy(inputs_[i], input.data(), kMaxBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(outputs_[i], 0xff, kMaxBytes), cudaSuccess);
  }
  SharedScratchDevice& device = scratch_pool().get(engines[0].handle()->device_id);
  const WithoutTheHandoffFlag without_flag(device);

  struct Gate {
    std::atomic<bool> open{false};
  } gate;
  ASSERT_EQ(
      cudaLaunchHostFunc(
          streams_[0],
          [](void* data) {
            // Bounded, so a case that fails before opening the gate cannot hang the stream.
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!static_cast<Gate*>(data)->open.load() && std::chrono::steady_clock::now() < give_up) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
          },
          &gate),
      cudaSuccess);
  Error first = Error::Internal;
  Error second = Error::Internal;
  // The held stream cannot finish a synchronize before the gate opens, so the first call reaches one only
  // after it has queued its enqueue and recorded it on the pool's event.
  std::atomic<bool> first_synchronizing{false};
  std::thread first_caller([&] {
    CudaCalls calls;
    calls.before_synchronize = [&](cudaStream_t stream) {
      if (stream == streams_[0]) {
        first_synchronizing.store(true);
      }
    };
    first = engines[0].run(inputs_[0], outputs_[0], 8, 32, streams_[0]);
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!first_synchronizing.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool first_reached_its_synchronize = first_synchronizing.load();
  std::atomic<bool> second_returned{false};
  size_t dropped_waits = 0;
  std::thread second_caller([&] {
    CudaCalls calls;
    calls.drop_stream_waits = true;
    second = engines[1].run(inputs_[1], outputs_[1], 8, 32, streams_[1]);
    dropped_waits = calls.dropped_stream_waits;
    second_returned.store(true);
  });
  bool lock_freed_while_held = false;
  const auto watch_until = std::chrono::steady_clock::now() + kWatchTheSecondCallFor;
  while (std::chrono::steady_clock::now() < watch_until && !second_returned.load()) {
    if (device.mu.try_lock()) {
      device.mu.unlock();
      lock_freed_while_held = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool second_returned_early = second_returned.load();
  gate.open.store(true);
  first_caller.join();
  second_caller.join();

  ASSERT_TRUE(first_reached_its_synchronize)
      << "the first call had not reached a synchronize on its held stream when the second call started";
  EXPECT_FALSE(lock_freed_while_held) << "the pool's lock was free while the call holding the scratch still ran";
  EXPECT_FALSE(second_returned_early)
      << "the second pooled call returned while the call before it was still held on the shared scratch";
  EXPECT_GT(dropped_waits, 0u) << "the second call queued no event wait, so dropping one tested nothing";
  EXPECT_EQ(first, Error::Ok);
  EXPECT_EQ(second, Error::Ok);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(cudaMemcpy(output.data(), outputs_[i], kMaxBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (float value : output) {
      ASSERT_FLOAT_EQ(value, 1.0f / 8);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The flag keeps the order when the driver drops the event wait. Here the second call's cudaStreamWaitEvent
// is dropped and the earlier pooled call is held on its stream, so only the flag can keep the second call,
// on another stream, from finishing first.
TEST_F(ExecutionGraphTest, TheFlagAloneOrdersPooledCallsWhenTheEventWaitIsDropped) {
  if (!device_can_wait_on_a_flag()) {
    GTEST_SKIP() << "This GPU has no 64-bit stream memory operations, so the pool has no flag";
  }
  constexpr std::chrono::milliseconds kWatchTheSecondCallFor{500};
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  LoadedGraphEngine engines[2];
  std::vector<float> input(kMaxElements, 1.0f), output(kMaxElements);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(engines[i].load(blob), Error::Ok);
    ASSERT_TRUE(engines[i].handle()->claims_pooled_scratch);
    ASSERT_EQ(cudaMemcpy(inputs_[i], input.data(), kMaxBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(outputs_[i], 0xff, kMaxBytes), cudaSuccess);
  }
  SharedScratchDevice& device = scratch_pool().get(engines[0].handle()->device_id);

  struct Gate {
    std::atomic<bool> open{false};
  } gate;
  ASSERT_EQ(
      cudaLaunchHostFunc(
          streams_[0],
          [](void* data) {
            // Bounded, so a case that fails before opening the gate cannot hang the stream.
            const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!static_cast<Gate*>(data)->open.load() && std::chrono::steady_clock::now() < give_up) {
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
          },
          &gate),
      cudaSuccess);
  Error first = Error::Internal;
  Error second = Error::Internal;
  size_t dropped = 0;
  std::thread first_caller([&] { first = engines[0].run(inputs_[0], outputs_[0], 8, 32, streams_[0]); });
  bool first_recorded = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!first_recorded && std::chrono::steady_clock::now() < deadline) {
    if (device.mu.try_lock()) {
      first_recorded = device.marker.pending;
      device.mu.unlock();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::thread second_caller([&] {
    CudaCalls calls;
    calls.drop_stream_waits = true;
    second = engines[1].run(inputs_[1], outputs_[1], 8, 32, streams_[1]);
    dropped = calls.dropped_stream_waits;
  });
  bool second_stream_busy = false;
  const auto busy_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!second_stream_busy && std::chrono::steady_clock::now() < busy_deadline) {
    second_stream_busy = cudaStreamQuery(streams_[1]) == cudaErrorNotReady;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  bool second_finished_early = false;
  const auto watch_until = std::chrono::steady_clock::now() + kWatchTheSecondCallFor;
  while (second_stream_busy && std::chrono::steady_clock::now() < watch_until) {
    if (cudaStreamQuery(streams_[1]) == cudaSuccess) {
      second_finished_early = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  gate.open.store(true);
  first_caller.join();
  second_caller.join();
  {
    std::lock_guard<std::mutex> lock(device.mu);
    ASSERT_FALSE(device.marker.event_wait_only) << "this GPU can wait on a flag, but the pool made none";
  }

  ASSERT_TRUE(first_recorded) << "the first call never recorded its enqueue, so there was nothing to wait for";
  ASSERT_TRUE(second_stream_busy) << "the second call never queued its work";
  EXPECT_GT(dropped, 0u) << "the second call made no event wait to drop";
  EXPECT_FALSE(second_finished_early)
      << "with its event wait dropped, the second pooled call finished while the call before it was still held";
  EXPECT_EQ(first, Error::Ok);
  EXPECT_EQ(second, Error::Ok);
  for (int i = 0; i < 2; ++i) {
    ASSERT_EQ(cudaMemcpy(output.data(), outputs_[i], kMaxBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (float value : output) {
      ASSERT_FLOAT_EQ(value, 1.0f / 8);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The sequence moves only once its write to the flag is queued. If a failed write moved it, the next pooled
// call would wait on the flag for a value never written.
TEST_F(ExecutionGraphTest, AFailedFlagWriteLeavesTheSequenceAtTheFlag) {
  if (!device_can_wait_on_a_flag()) {
    GTEST_SKIP() << "This GPU has no 64-bit stream memory operations, so the pool has no flag";
  }
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  LoadedGraphEngine engine;
  ASSERT_EQ(engine.load(blob), Error::Ok);
  ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
  ASSERT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::Ok);
  SharedScratchDevice& device = scratch_pool().get(engine.handle()->device_id);
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(device.mu);
    ASSERT_NE(device.marker.flag, nullptr);
    sequence = device.marker.sequence;
  }
  fail_flag_writes = 1;
  EXPECT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::InvalidState);
  EXPECT_EQ(fail_flag_writes.load(), 0) << "the call made no write to fail";
  fail_flag_writes = 0;
  {
    std::lock_guard<std::mutex> lock(device.mu);
    ASSERT_EQ(device.marker.sequence, sequence) << "the next pooled call would wait for a value never written";
  }
  EXPECT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::Ok);
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// A pooled engine records while it holds its device's pool lock, so it must not wait for the capture
// lock that a load, a destroy, or another engine's recording on any device holds. It runs that call
// plainly and records later.
TEST_F(ExecutionGraphReplayTest, PooledRecordingDoesNotWaitForTheCaptureLock) {
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
    std::vector<float> input(8 * 32, 1.0f), output(input.size(), -1.0f);
    ASSERT_EQ(engine.run(input.data(), output.data(), 8, 32, streams_[0]), Error::Ok);
    // A new input changes the result, so a call that skipped its launch would return the old one.
    input[0] = 2.0f;
    output.assign(output.size(), -1.0f);
    std::promise<void> held, release;
    auto released = release.get_future();
    std::thread holder([&] {
      const std::shared_lock<std::shared_mutex> lock(cuda_graph_capture_mutex());
      held.set_value();
      // Bounded, so a pooled call that does wait fails this test instead of hanging it.
      released.wait_for(std::chrono::seconds(10));
    });
    held.get_future().wait();
    Error error = Error::Ok;
    size_t captures_while_held = 0;
    const auto start = std::chrono::steady_clock::now();
    {
      CudaCalls calls;
      error = engine.run(input.data(), output.data(), 8, 32, streams_[0]);
      captures_while_held = calls.captures.size();
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    const bool captured_while_held = engine.is_captured();
    release.set_value();
    holder.join();
    EXPECT_EQ(error, Error::Ok);
    EXPECT_LT(waited, std::chrono::seconds(5)) << "the pooled call waited for the capture lock";
    EXPECT_EQ(captures_while_held, 0u);
    EXPECT_FALSE(captured_while_held);
    const std::vector<float> contended = output;
    EXPECT_NE(contended[0], 1.0f / 8) << "the pooled call returned the previous call's result";
    CudaCalls calls;
    for (int i = 0; i < 2; ++i) {
      output.assign(output.size(), -1.0f);
      ASSERT_EQ(engine.run(input.data(), output.data(), 8, 32, streams_[0]), Error::Ok);
      for (size_t j = 0; j < output.size(); ++j) {
        ASSERT_FLOAT_EQ(output[j], contended[j]);
      }
    }
    EXPECT_TRUE(engine.is_captured());
    EXPECT_EQ(calls.captures.size(), 1u);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(2, streams_[0]));
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// With pooled scratch allowed to replay, aliased outputs are the guard that keeps such an engine plain.
TEST_F(ExecutionGraphReplayTest, PooledEngineWithAliasedOutputsStaysPlain) {
  const auto blob = build_scratch_blob(8, 32, true);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
    ASSERT_GT(engine.handle()->num_aliased_outputs, 0u);
    EXPECT_EQ(engine.handle()->execution_graph, nullptr);
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

// The plain path can fail after the claim too, so it also has to drain before the pool's lock is
// released. A failed record of the enqueue on the pool's event fails the call after its enqueue.
TEST_F(ExecutionGraphTest, AFailedPooledPlainCallDrainsBeforeReleasingThePool) {
  const auto blob = build_scratch_blob(8, 32);
  ASSERT_FALSE(blob.empty());
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, false)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  {
    LoadedGraphEngine engine;
    ASSERT_EQ(engine.load(blob), Error::Ok);
    ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
    ASSERT_EQ(engine.handle()->execution_graph, nullptr);
    std::vector<float> input(kMaxElements, 1.0f), output(kMaxElements, -1.0f);
    ASSERT_EQ(cudaMemcpy(inputs_[0], input.data(), kMaxBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::Ok);
    SharedScratchDevice& dev = scratch_pool().get(engine.handle()->device_id);
    std::vector<bool> held_at_drain;
    {
      CudaCalls calls;
      calls.fail_event_record_call = 1;
      calls.before_synchronize = [&](cudaStream_t stream) {
        if (stream != streams_[0]) {
          return;
        }
        // try_lock on the thread that owns the lock is undefined, so the probe runs on another thread.
        held_at_drain.push_back(!std::async(std::launch::async, [&] {
                                   const bool lock_free = dev.mu.try_lock();
                                   if (lock_free) {
                                     dev.mu.unlock();
                                   }
                                   return lock_free;
                                 }).get());
      };
      EXPECT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::InvalidState);
      EXPECT_EQ(calls.event_record_calls, 1u);
    }
    ASSERT_FALSE(held_at_drain.empty()) << "the failed call never drained its stream";
    EXPECT_EQ(held_at_drain, std::vector<bool>(held_at_drain.size(), true))
        << "the pool's lock was released before the failed call's stream was drained";
    ASSERT_EQ(cudaMemset(outputs_[0], 0xff, kMaxBytes), cudaSuccess);
    ASSERT_EQ(engine.run(inputs_[0], outputs_[0], 8, 32, streams_[0]), Error::Ok);
    ASSERT_EQ(cudaMemcpy(output.data(), outputs_[0], kMaxBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (float value : output) {
      ASSERT_FLOAT_EQ(value, 1.0f / 8);
    }
  }
  EXPECT_TRUE(reset_shared_scratch_pool_for_testing());
}

TEST_F(ExecutionGraphTest, MissingMemoryPoolSupportDisablesReplayAtInitialization) {
  LoadedGraphEngine plain, no_pools;
  ASSERT_EQ(set_graphs(false), Error::Ok);
  ASSERT_EQ(plain.load(blob_), Error::Ok);
  ASSERT_EQ(set_graphs(true), Error::Ok);
  {
    CudaCalls initialization;
    initialization.disable_memory_pools = true;
    ASSERT_EQ(no_pools.load(blob_), Error::Ok);
    EXPECT_EQ(initialization.memory_pool_queries, 1u);
  }
  EXPECT_EQ(no_pools.handle()->execution_graph, nullptr);
  CudaCalls calls;
  for (int i = 0; i < 5; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, no_pools, i % 2, 2, 8, i, streams_[0]));
    EXPECT_EQ(no_pools.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
    EXPECT_EQ(no_pools.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
  }
  EXPECT_EQ(calls.malloc_calls, 0u);
  EXPECT_TRUE(calls.captures.empty());
  EXPECT_TRUE(calls.launches.empty());
}

TEST_F(ExecutionGraphReplayTest, CaptureAllowsConcurrentLegacyStreamWork) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  CudaCalls calls;
  size_t legacy_operations = 0;
  unsigned int flags = 0;
  // The capture stream is destroyed right after the recording, so it is asked about before that.
  calls.after_capture = [&](cudaGraph_t) { EXPECT_EQ(cudaStreamGetFlags(calls.captures.back(), &flags), cudaSuccess); };
  calls.during_capture = [&](cudaStream_t capture_stream) {
    EXPECT_NE(capture_stream, streams_[0]);
    std::thread legacy([&] {
      ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
      for (int i = 0; i < 32; ++i) {
        ASSERT_EQ(cudaMemsetAsync(inputs_[1], 0, kMaxBytes, cudaStreamLegacy), cudaSuccess);
        float value = -1.0f;
        ASSERT_EQ(cudaMemcpy(&value, inputs_[1], sizeof(value), cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_FLOAT_EQ(value, 0.0f);
        ++legacy_operations;
      }
    });
    legacy.join();
  };
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 2, streams_[0]));
  EXPECT_EQ(legacy_operations, 32u);
  ASSERT_EQ(calls.captures.size(), 1u);
  EXPECT_NE(flags & cudaStreamNonBlocking, 0u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, CaptureDoesNotAbsorbAnotherEngineOnTheCallerStream) {
  LoadedGraphEngine other, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(other, graph));
  ASSERT_NO_FATAL_FAILURE(compare(other, graph, 0, 2, 8, 1, streams_[0]));
  const std::vector<float> other_input(16, 50.0f);
  std::vector<float> other_output(16, -7.0f);
  ASSERT_EQ(cudaMemcpyAsync(inputs_[1], other_input.data(), 64, cudaMemcpyHostToDevice, streams_[0]), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(outputs_[1], other_output.data(), 64, cudaMemcpyHostToDevice, streams_[0]), cudaSuccess);
  CudaCalls calls;
  size_t overlaps = 0;
  calls.during_capture = [&](cudaStream_t capture_stream) {
    EXPECT_NE(capture_stream, streams_[0]);
    std::thread worker([&] {
      ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
      ASSERT_EQ(other.run(inputs_[1], outputs_[1], 2, 8, streams_[0]), Error::Ok);
      ASSERT_EQ(cudaMemcpy(other_output.data(), outputs_[1], 64, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(other_output, std::vector<float>(16, 101.0f));
      ++overlaps;
    });
    worker.join();
  };
  ASSERT_NO_FATAL_FAILURE(compare(other, graph, 0, 2, 8, 2, streams_[0]));
  EXPECT_EQ(overlaps, 1u);
  EXPECT_EQ(calls.captures.size(), 1u);
  const std::vector<float> sentinel(16, -7.0f);
  ASSERT_EQ(cudaMemcpyAsync(outputs_[1], sentinel.data(), 64, cudaMemcpyHostToDevice, streams_[0]), cudaSuccess);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(other, graph, 0, 2, 8, i + 3, streams_[0]));
    ASSERT_EQ(cudaMemcpy(other_output.data(), outputs_[1], 64, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(other_output, sentinel);
  }
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(4, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, TwoGraphEnginesCaptureIndependentlyOnOneCallerStream) {
  LoadedGraphEngine plain, first, second;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, first));
  ASSERT_EQ(second.load(blob_), Error::Ok);
  ASSERT_NO_FATAL_FAILURE(compare(plain, first, 0, 2, 8, 1, streams_[0]));
  const std::vector<float> input(16, 50.0f);
  ASSERT_EQ(cudaMemcpyAsync(inputs_[1], input.data(), 64, cudaMemcpyHostToDevice, streams_[0]), cudaSuccess);
  ASSERT_EQ(second.run(inputs_[1], outputs_[1], 2, 8, streams_[0]), Error::Ok);
  CudaCalls calls;
  std::promise<void> started, finished;
  auto start = started.get_future();
  auto finish = finished.get_future();
  std::thread worker;
  calls.during_capture = [&](cudaStream_t) {
    worker = std::thread([&] {
      EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
      started.set_value();
      CudaCalls second_calls;
      EXPECT_EQ(second.run(inputs_[1], outputs_[1], 2, 8, streams_[0]), Error::Ok);
      EXPECT_EQ(second_calls.captures.size(), 1u);
      if (!second_calls.captures.empty()) {
        EXPECT_NE(second_calls.captures.front(), streams_[0]);
      }
      EXPECT_EQ(second_calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
      std::vector<float> output(16);
      EXPECT_EQ(cudaMemcpy(output.data(), outputs_[1], 64, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(output, std::vector<float>(16, 101.0f));
      finished.set_value();
    });
    EXPECT_EQ(start.wait_for(std::chrono::seconds(30)), std::future_status::ready);
    EXPECT_EQ(finish.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  };
  compare(plain, first, 0, 2, 8, 2, streams_[0]);
  if (worker.joinable()) {
    worker.join();
  }
  EXPECT_EQ(finish.wait_for(std::chrono::seconds(0)), std::future_status::ready);
  EXPECT_EQ(calls.captures.size(), 1u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, CaptureSerializesWithBackendLoadsAndDestruction) {
  for (bool share : {false, true}) {
    SCOPED_TRACE(share);
    BackendOption options[] = {option(kSharedEnginesKey, share), option(kGraphOption, false)};
    for (bool destroy : {false, true}) {
      SCOPED_TRACE(destroy);
      LoadedGraphEngine graph;
      ASSERT_NO_FATAL_FAILURE(load_graph(graph));
      ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, 1));
      auto retiring = std::make_unique<LoadedGraphEngine>();
      ASSERT_EQ(retiring->load(blob_, {}, {options, 2}), Error::Ok);
      std::promise<void> started, finished;
      auto start = started.get_future();
      auto finish = finished.get_future();
      std::thread worker;
      CudaCalls calls;
      calls.during_capture = [&](cudaStream_t) {
        worker = std::thread([&] {
          EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
          started.set_value();
          if (destroy) {
            retiring.reset();
          } else {
            LoadedGraphEngine loaded;
            EXPECT_EQ(loaded.load(blob_, {}, {options, 2}), Error::Ok);
          }
          finished.set_value();
        });
        EXPECT_EQ(start.wait_for(std::chrono::seconds(30)), std::future_status::ready);
        EXPECT_EQ(finish.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout)
            << "backend load or destruction overlapped capture";
      };
      compare_default(graph, 0, 2, 8, 2);
      if (worker.joinable()) {
        worker.join();
      }
      ASSERT_EQ(finish.wait_for(std::chrono::seconds(0)), std::future_status::ready);
      EXPECT_TRUE(graph.is_captured());
      EXPECT_EQ(calls.captures.size(), 1u);
    }

    LoadedGraphEngine graph;
    ASSERT_NO_FATAL_FAILURE(load_graph(graph));
    const std::vector<float> input(16, 3.0f);
    ASSERT_EQ(cudaMemcpy(inputs_[0], input.data(), 64, cudaMemcpyHostToDevice), cudaSuccess);
    std::atomic<bool> started{false};
    std::atomic<int> loads{0};
    std::thread churn([&] {
      EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
      started.store(true);
      for (int i = 0; i < 200; ++i) {
        LoadedGraphEngine loaded;
        EXPECT_EQ(loaded.load(blob_, {}, {options, 2}), Error::Ok);
        ++loads;
      }
    });
    while (!started.load()) {
      std::this_thread::yield();
    }
    CudaCalls calls;
    for (int i = 0; i < 2000; ++i) {
      const bool square = (i / 2) % 2 != 0;
      EXPECT_EQ(graph.run(inputs_[0], outputs_[0], square ? 4 : 2, square ? 4 : 8, streams_[0]), Error::Ok);
      EXPECT_EQ(graph.is_captured(), i % 2 != 0);
    }
    churn.join();
    EXPECT_EQ(loads.load(), 200);
    EXPECT_EQ(calls.captures.size(), 1000u);
    std::vector<float> output(16);
    ASSERT_EQ(cudaMemcpy(output.data(), outputs_[0], 64, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(output, std::vector<float>(16, 7.0f));
  }
}

TEST_F(ExecutionGraphReplayTest, AllocationFailureUsesCallerBindingsAndSuppressesRetries) {
  for (size_t failed_allocation : {1u, 2u}) {
    for (bool growth : {false, true}) {
      SCOPED_TRACE(failed_allocation);
      SCOPED_TRACE(growth);
      LoadedGraphEngine plain, graph;
      ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
      if (growth) {
        // Staged under other shapes, so the first call of the loop, a warm-up, grows the buffers.
        ASSERT_NO_FATAL_FAILURE(stage(plain, graph, 2, 8, streams_[0]));
      } else {
        // A warm-up, a recording and the first move. The second move, which allocates the buffers, is the
        // first call of the loop after this one.
        for (int slot : {0, 0, 1}) {
          ASSERT_NO_FATAL_FAILURE(compare(plain, graph, slot, 8, 32, slot, streams_[0]));
        }
      }
      CudaCalls calls;
      calls.fail_malloc_call = failed_allocation;
      for (int i = 0; i < 5; ++i) {
        ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 8, 32, i + 3, streams_[0]));
        EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
        EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
        EXPECT_FALSE(graph.is_captured());
      }
      EXPECT_EQ(calls.malloc_calls, failed_allocation);
      EXPECT_TRUE(calls.captures.empty());
      EXPECT_TRUE(calls.launches.empty());
      EXPECT_EQ(calls.free_calls, 0u);
      EXPECT_EQ(calls.retirements.size(), (growth ? 2 : 0) + failed_allocation - 1);
    }
  }
}

TEST_F(ExecutionGraphReplayTest, CaptureFailureRestoresCallerBindingsAndSuppressesRetries) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  {
    CudaCalls calls;
    calls.fail_captures = 3;
    // The same buffers until recording gives up, so every recording goes on the caller's addresses.
    const int slots[] = {0, 0, 0, 1, 0, 1};
    for (int i = 0; i < 6; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, slots[i], 2, 8, i + 2, streams_[0]));
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[slots[i]]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[slots[i]]);
      EXPECT_FALSE(graph.is_captured());
    }
    EXPECT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
    EXPECT_EQ(calls.malloc_calls, 0u);
    EXPECT_EQ(calls.free_calls, 0u);
    EXPECT_TRUE(calls.retirements.empty());
  }
  CudaCalls recovery;
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 4, 4, i + 8, streams_[0]));
  }
  EXPECT_EQ(recovery.captures.size(), 1u);
  EXPECT_EQ(recovery.launches, std::vector<cudaStream_t>(2, streams_[0]));
}

// The same on the graph's own buffers: a failed recording there leaves the context on those buffers,
// so the call has to bind the caller's again before it runs, and the third failure frees them.
TEST_F(ExecutionGraphReplayTest, StagedCaptureFailureRestoresCallerBindingsAndReleasesBuffers) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i : {0, 0, 1}) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  {
    CudaCalls calls;
    calls.fail_captures = 3;
    // The first call moves the buffers a second time and tries to record on the graph's own buffers.
    for (int i = 0; i < 4; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i + 3, streams_[0]));
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
      EXPECT_FALSE(graph.is_captured());
    }
    EXPECT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
    EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, streams_[0]));
  }
}

TEST_F(ExecutionGraphReplayTest, GrowthRetiresBuffersOnTheCallerStreamWithoutCudaFree) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(stage(plain, graph, 2, 8, streams_[0]));
  // The buffers have moved twice, so a new shape stays on the graph's buffers, and its warm-up grows them.
  CudaCalls calls;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 8, 32, 3, streams_[0]));
  EXPECT_EQ(calls.malloc_calls, 2u);
  EXPECT_EQ(calls.free_calls, 0u);
  EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, streams_[0]));
  EXPECT_TRUE(calls.launches.empty());
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 1, 8, 32, 4, streams_[0]));
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, GraphLaunchReturnsTheCudaError) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, streams_[0]));
  }
  CudaCalls calls;
  calls.launch_error = cudaErrorMemoryAllocation;
  EXPECT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, streams_[0]), Error::MemoryAllocationFailed);
  calls.launch_error = cudaErrorInvalidValue;
  EXPECT_EQ(graph.run(inputs_[0], outputs_[0], 2, 8, streams_[0]), Error::Internal);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(2, streams_[0]));
}

// Host buffers off a 256-byte boundary always go through the delegate's own device buffers, whose
// addresses stay put while the caller alternates two sets.
TEST_F(ExecutionGraphReplayTest, HostBuffersStayCurrentOnReplay) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  HostFloats inputs[2] = {HostFloats(16), HostFloats(16)};
  HostFloats outputs[2] = {HostFloats(16), HostFloats(16)};
  std::vector<float> expected(16);
  for (int i = 0; i < 5; ++i) {
    SCOPED_TRACE(i);
    const int slot = i % 2;
    for (size_t j = 0; j < 16; ++j) {
      inputs[slot].data[j] = static_cast<float>(i * 16 + j);
      outputs[slot].data[j] = -1000.0f;
    }
    ASSERT_EQ(plain.run(inputs[slot].data, expected.data(), 2, 8, streams_[0]), Error::Ok);
    ASSERT_EQ(graph.run(inputs[slot].data, outputs[slot].data, 2, 8, streams_[0]), Error::Ok);
    EXPECT_EQ(graph.is_captured(), i >= 1);
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_FLOAT_EQ(outputs[slot].data[j], expected[j]);
      ASSERT_FLOAT_EQ(outputs[slot].data[j], inputs[slot].data[j] * 2.0f + 1.0f);
    }
  }
}

TEST_F(ExecutionGraphReplayTest, RuntimeOptionOnlyAffectsSubsequentlyLoadedEngines) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_EQ(set_graphs(false), Error::Ok);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, streams_[0]));
  }
  EXPECT_TRUE(graph.is_captured());
  LoadedGraphEngine later;
  ASSERT_EQ(later.load(blob_), Error::Ok);
  EXPECT_EQ(later.handle()->execution_graph, nullptr);
  ASSERT_EQ(set_graphs(true), Error::Ok);
  EXPECT_EQ(plain.handle()->execution_graph, nullptr);
  EXPECT_EQ(later.handle()->execution_graph, nullptr);
}

TEST_F(ExecutionGraphReplayTest, InvalidOptionTypesLeaveTheWholeRequestUnapplied) {
  for (bool enabled : {false, true}) {
    BackendOption initial[] = {option(kGraphOption, enabled), option(kScratchOption, false)};
    ASSERT_EQ(set_options({initial, 2}), Error::Ok);
    BackendOption wrong_graph = option(kGraphOption, !enabled);
    wrong_graph.value = enabled ? 0 : 1;
    BackendOption request[] = {option(kScratchOption, true), option(kGraphOption, !enabled), wrong_graph};
    EXPECT_EQ(set_options({request, 3}), Error::InvalidArgument);
    LoadedGraphEngine after_graph_error;
    ASSERT_EQ(after_graph_error.load(blob_), Error::Ok);
    EXPECT_EQ(after_graph_error.handle()->execution_graph != nullptr, enabled);
    EXPECT_FALSE(after_graph_error.handle()->shared_scratch);

    BackendOption wrong_scratch = option(kScratchOption, true);
    wrong_scratch.value = 1;
    BackendOption other_request[] = {option(kGraphOption, !enabled), wrong_scratch};
    EXPECT_EQ(set_options({other_request, 2}), Error::InvalidArgument);
    LoadedGraphEngine after_scratch_error;
    ASSERT_EQ(after_scratch_error.load(blob_), Error::Ok);
    EXPECT_EQ(after_scratch_error.handle()->execution_graph != nullptr, enabled);
    EXPECT_FALSE(after_scratch_error.handle()->shared_scratch);
  }
}

TEST_F(ExecutionGraphTest, UserAliasesKeepCallerStorageWithoutReplayAllocations) {
  LoadedGraphEngine graph;
  ASSERT_EQ(graph.load(alias_blob_), Error::Ok);
  EXPECT_EQ(graph.handle()->execution_graph, nullptr);
  ASSERT_EQ(graph.handle()->num_aliased_outputs, 1u);
  constexpr size_t count = 16;
  for (int run = 0; run < 5; ++run) {
    const int slot = run < 2 ? 0 : (run + 1) % 2;
    std::vector<float> input(count, static_cast<float>(run + 1));
    ASSERT_EQ(
        cudaMemcpyAsync(inputs_[slot], input.data(), count * sizeof(float), cudaMemcpyHostToDevice, streams_[0]),
        cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(outputs_[slot], 0xff, kMaxBytes, streams_[0]), cudaSuccess);
    {
      CudaCalls calls;
      ASSERT_EQ(graph.run(inputs_[slot], outputs_[slot], 2, 8, streams_[0]), Error::Ok);
      EXPECT_EQ(calls.malloc_calls, 0u);
      EXPECT_EQ(calls.memcpy_calls, 1u);
      EXPECT_TRUE(calls.captures.empty());
      EXPECT_TRUE(calls.launches.empty());
    }
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[slot]);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), inputs_[slot]);
    std::vector<float> mutated(count), reflected(count);
    ASSERT_EQ(cudaMemcpy(mutated.data(), inputs_[slot], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(reflected.data(), outputs_[slot], count * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_FLOAT_EQ(mutated[i], input[i] * 2.0f + 1.0f);
    }
    EXPECT_EQ(reflected, mutated);
  }
}

TEST_F(ExecutionGraphTest, KvAliasesKeepCallerStorageForThreadedAndElidedOutputs) {
  ASSERT_FALSE(kv_blob_.empty());
  LoadedGraphEngine graph;
  ASSERT_EQ(graph.load(kv_blob_), Error::Ok);
  ASSERT_EQ(graph.handle()->execution_graph, nullptr);
  ASSERT_EQ(graph.handle()->num_aliased_outputs, 1u);
  for (bool elided : {false, true}) {
    for (int run = 0; run < 5; ++run) {
      const int slot = run % 2;
      std::vector<float> cache(16, static_cast<float>(run));
      ASSERT_EQ(cudaMemcpyAsync(inputs_[slot], cache.data(), 64, cudaMemcpyHostToDevice, streams_[0]), cudaSuccess);
      const float update[] = {9, 10, 11, 12};
      void* update_pointer = outputs_[1 - slot];
      ASSERT_EQ(
          cudaMemcpyAsync(update_pointer, update, sizeof(update), cudaMemcpyHostToDevice, streams_[0]), cudaSuccess);
      SizesType sizes[] = {1, 1, 4, 4};
      SizesType update_sizes[] = {1, 1, 1, 4};
      TensorImpl cache_impl(ScalarType::Float, 4, sizes, inputs_[slot]);
      TensorImpl update_impl(ScalarType::Float, 4, update_sizes, update_pointer);
      TensorImpl reflected_impl(ScalarType::Float, 4, sizes, reference_output_);
      TensorImpl output_impl(ScalarType::Float, 4, update_sizes, outputs_[slot]);
      EValue cache_value{Tensor(&cache_impl)};
      EValue update_value{Tensor(&update_impl)};
      EValue reflected_value{Tensor(&reflected_impl)};
      EValue output_value{Tensor(&output_impl)};
      EValue* threaded_args[] = {&cache_value, &update_value, &reflected_value, &output_value};
      EValue* elided_args[] = {&cache_value, &update_value, &output_value};
      {
        CudaCalls calls;
        ASSERT_EQ(
            graph.run(elided ? Span<EValue*>(elided_args, 3) : Span<EValue*>(threaded_args, 4), streams_[0]),
            Error::Ok);
        EXPECT_EQ(calls.malloc_calls, 0u);
        EXPECT_EQ(calls.memcpy_calls, elided ? 0u : 1u);
        EXPECT_TRUE(calls.captures.empty());
        EXPECT_TRUE(calls.launches.empty());
      }
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[slot]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), inputs_[slot]);
      std::vector<float> actual(16), output(4);
      ASSERT_EQ(cudaMemcpy(actual.data(), inputs_[slot], 64, cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(output.data(), outputs_[slot], 16, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(output, std::vector<float>({10, 11, 12, 13}));
      for (size_t i = 0; i < 16; ++i) {
        const float expected = i >= 4 && i < 8 ? static_cast<float>(i + 5) : cache[i];
        EXPECT_FLOAT_EQ(actual[i], expected);
      }
      if (!elided) {
        std::vector<float> reflected(16);
        ASSERT_EQ(cudaMemcpy(reflected.data(), reference_output_, 64, cudaMemcpyDeviceToHost), cudaSuccess);
        EXPECT_EQ(reflected, actual);
      }
    }
  }
}

TEST_F(ExecutionGraphReplayTest, ExecuteAndDestroyRestoreAnotherCurrentDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices to observe device restoration";
  }
  LoadedGraphEngine plain;
  auto graph = std::make_unique<LoadedGraphEngine>();
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, *graph));
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    const std::vector<float> input(16, static_cast<float>(i));
    ASSERT_EQ(
        cudaMemcpyAsync(inputs_[i % 2], input.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, streams_[0]),
        cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(outputs_[i % 2], 0xff, kMaxBytes, streams_[0]), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(graph->run(inputs_[i % 2], outputs_[i % 2], 2, 8, streams_[0]), Error::Ok);
    int current = -1;
    ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
    EXPECT_EQ(current, 1);
    EXPECT_EQ(graph->is_captured(), i >= 1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams_[0]), cudaSuccess);
    std::vector<float> actual(16);
    ASSERT_EQ(cudaMemcpy(actual.data(), outputs_[i % 2], 16 * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, std::vector<float>(16, static_cast<float>(i * 2 + 1)));
  }
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  graph.reset();
  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 1);
}

// A buffer on another device than the engine's is refused before the engine runs, as an input, as an output,
// and as the buffer an aliased output is copied into.
TEST_F(ExecutionGraphTest, ABufferOnAnotherDeviceIsRefused) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices for a buffer on another device";
  }
  void* other = nullptr;
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&other, kMaxBytes), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  LoadedGraphEngine engine, alias;
  ASSERT_EQ(engine.load(blob_), Error::Ok);
  ASSERT_EQ(alias.load(alias_blob_), Error::Ok);
  EXPECT_EQ(engine.run(other, outputs_[0], 2, 8, streams_[0]), Error::InvalidArgument);
  EXPECT_EQ(engine.run(inputs_[0], other, 2, 8, streams_[0]), Error::InvalidArgument);
  EXPECT_EQ(alias.run(inputs_[0], other, 2, 8, streams_[0]), Error::InvalidArgument);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  EXPECT_EQ(cudaFree(other), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
