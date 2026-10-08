/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "execution_graph_cuda_wrappers.h"
#include "torch_tensorrt/executorch/ExecutionGraph.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <cudaTypedefs.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
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

std::vector<uint8_t> build_blob(bool alias) {
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
  auto* profile = builder->createOptimizationProfile();
  if (!profile || !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, 2}) ||
      !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{2, 8}) ||
      !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{8, 32}) ||
      config->addOptimizationProfile(profile) < 0) {
    return {};
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
std::vector<uint8_t> build_scratch_blob() {
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
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{8, 32});
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

TEST_F(ExecutionGraphReplayTest, CapturesSecondCallAndReplaysChangedAddressesAndValues) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  CudaCalls calls;
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  EXPECT_TRUE(calls.launches.empty());
  EXPECT_FALSE(graph.is_captured());
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  ASSERT_TRUE(graph.is_captured());
  const void* stable_input = graph.handle()->exec_ctx->getTensorAddress("input_0");
  const void* stable_output = graph.handle()->exec_ctx->getTensorAddress("output_0");
  for (int i = 0; i < 6; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, (i + 1) % 2, 2, 8, i + 2, streams_[0]));
    ASSERT_TRUE(graph.is_captured());
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), stable_input);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), stable_output);
  }
  ASSERT_EQ(calls.captures.size(), 1u);
  EXPECT_NE(calls.captures.front(), streams_[0]);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(7, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamReplaysHostPointers) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  std::vector<float> inputs[2] = {std::vector<float>(16), std::vector<float>(16)};
  std::vector<float> outputs[2] = {std::vector<float>(16), std::vector<float>(16)};
  std::vector<float> expected(16);
  ASSERT_NE(inputs[0].data(), inputs[1].data());
  ASSERT_NE(outputs[0].data(), outputs[1].data());
  size_t launches = 0;
  size_t captures = 0;
  for (int i = 0; i < 6; ++i) {
    SCOPED_TRACE(i);
    const int slot = i < 3 ? 0 : i % 2;
    for (size_t j = 0; j < 16; ++j) {
      inputs[slot][j] = static_cast<float>(i * 16 + j);
      outputs[slot][j] = -1000.0f;
    }
    ASSERT_FALSE(getCallerStream().has_value());
    ASSERT_EQ(plain.run(inputs[slot].data(), expected.data(), 2, 8), Error::Ok);
    CudaCalls calls;
    ASSERT_EQ(graph.run(inputs[slot].data(), outputs[slot].data(), 2, 8), Error::Ok);
    EXPECT_FALSE(getCallerStream().has_value());
    EXPECT_EQ(outputs[slot], expected);
    for (size_t j = 0; j < 16; ++j) {
      EXPECT_FLOAT_EQ(outputs[slot][j], inputs[slot][j] * 2.0f + 1.0f);
    }
    ASSERT_FALSE(calls.operations.empty());
    EXPECT_EQ(calls.operations.back().stream, cudaStreamPerThread);
    EXPECT_EQ(graph.is_captured(), i >= 1);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(i == 0 ? 0 : 1, cudaStreamPerThread));
    captures += calls.captures.size();
    launches += calls.launches.size();
  }
  EXPECT_EQ(captures, 1u);
  EXPECT_EQ(launches, 5u);
}

TEST_F(ExecutionGraphReplayTest, NoCallerStreamReplaysDevicePointers) {
  LoadedGraphEngine graph;
  ASSERT_NO_FATAL_FAILURE(load_graph(graph));
  const void* stable_input = nullptr;
  const void* stable_output = nullptr;
  size_t launches = 0;
  size_t captures = 0;
  for (int i = 0; i < 6; ++i) {
    SCOPED_TRACE(i);
    CudaCalls calls;
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, i < 3 ? 0 : i % 2, 2, 8, i));
    ASSERT_FALSE(calls.operations.empty());
    EXPECT_EQ(calls.operations.back().stream, cudaStreamPerThread);
    if (i == 0) {
      stable_input = graph.handle()->exec_ctx->getTensorAddress("input_0");
      stable_output = graph.handle()->exec_ctx->getTensorAddress("output_0");
      EXPECT_NE(stable_input, inputs_[0]);
      EXPECT_NE(stable_output, outputs_[0]);
    }
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), stable_input);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), stable_output);
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
        for (int i = 0; i < 2; ++i) {
          ASSERT_NO_FATAL_FAILURE(compare_default(graph, 0, 2, 8, i));
        }
        ASSERT_TRUE(graph.is_captured());
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
    for (int i = 0; i < 6; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare_default(graph, i % 2, 2, 8, i + 1));
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
      EXPECT_FALSE(graph.is_captured());
    }
    ASSERT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
    EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, cudaStreamPerThread));
    EXPECT_EQ(calls.malloc_calls, 0u);
    EXPECT_EQ(calls.free_calls, 0u);
  }
  CudaCalls recovery;
  // The new shape re-arms replay: warm-up, then capture and launch, then replays.
  for (int i = 0; i < 4; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare_default(graph, i % 2, 4, 4, i + 6));
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
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i + 2, streams_[0]));
    EXPECT_FALSE(graph.is_captured());
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
  }
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i + 4, streams_[0]));
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
  ASSERT_EQ(cudaMemcpy(inputs_[1], input.data(), 64, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(outputs_[1], 0xff, kMaxBytes), cudaSuccess);
  size_t mallocs = 0, launches = 0;
  Error result = Error::Internal;
  // No CUDA call may run on this thread before execute, so it starts with no current context.
  std::thread fresh([&] {
    CudaCalls calls;
    result = graph.run(inputs_[1], outputs_[1], 2, 8);
    mallocs = calls.malloc_calls;
    launches = calls.launches.size();
  });
  fresh.join();
  EXPECT_EQ(result, Error::Ok);
  EXPECT_EQ(mallocs, 0u);
  EXPECT_EQ(launches, 1u);
  EXPECT_TRUE(graph.is_captured());
  std::vector<float> output(16);
  ASSERT_EQ(cudaMemcpy(output.data(), outputs_[1], 64, cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i], input[i] * 2.0f + 1.0f);
  }
}

TEST_F(ExecutionGraphReplayTest, DestroyFreesReplayBuffers) {
  auto graph = std::make_unique<LoadedGraphEngine>();
  ASSERT_NO_FATAL_FAILURE(load_graph(*graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare_default(*graph, 0, 2, 8, i));
  }
  ASSERT_TRUE(graph->is_captured());
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
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 4, 4, i + 6, streams_[0]));
  }
  EXPECT_EQ(recovery.captures.size(), 1u);
  EXPECT_EQ(recovery.launches, std::vector<cudaStream_t>(2, streams_[0]));
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

TEST_F(ExecutionGraphReplayTest, StreamChangesKeepTheGraph) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 1, streams_[0]));
  ASSERT_FALSE(graph.is_captured());
  CudaCalls calls;
  int seed = 2;
  for (cudaStream_t stream : {streams_[0], streams_[1], streams_[0], streams_[1]}) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, seed % 2, 2, 8, seed, stream));
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
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, stream));
      EXPECT_EQ(graph.is_captured(), i >= 1);
    }
    ASSERT_EQ(calls.captures.size(), 1u);
    EXPECT_NE(calls.captures.front(), stream);
    EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(5, stream));
  }
  EXPECT_EQ(cudaStreamDestroy(blocking), cudaSuccess);
}

TEST_F(ExecutionGraphReplayTest, GreenContextStreamRunsPlainAndKeepsTheGraph) {
  const auto get_resource = driver_entry<PFN_cuDeviceGetDevResource_v12040>("cuDeviceGetDevResource", 12040);
  const auto generate_desc = driver_entry<PFN_cuDevResourceGenerateDesc_v12040>("cuDevResourceGenerateDesc", 12040);
  const auto create_green = driver_entry<PFN_cuGreenCtxCreate_v12040>("cuGreenCtxCreate", 12040);
  const auto destroy_green = driver_entry<PFN_cuGreenCtxDestroy_v12040>("cuGreenCtxDestroy", 12040);
  const auto create_stream = driver_entry<PFN_cuGreenCtxStreamCreate_v12050>("cuGreenCtxStreamCreate", 12050);
  const auto get_context = driver_entry<PFN_cuStreamGetCtx_v12050>("cuStreamGetCtx", 12050);
  ASSERT_NE(get_resource, nullptr);
  ASSERT_NE(generate_desc, nullptr);
  ASSERT_NE(create_green, nullptr);
  ASSERT_NE(destroy_green, nullptr);
  ASSERT_NE(create_stream, nullptr);
  ASSERT_NE(get_context, nullptr);

  struct GreenStream {
    CUgreenCtx context = nullptr;
    cudaStream_t stream = nullptr;
    PFN_cuGreenCtxDestroy_v12040 destroy;
    ~GreenStream() {
      if (stream) {
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      }
      if (context) {
        EXPECT_EQ(destroy(context), CUDA_SUCCESS);
      }
    }
  } green{nullptr, nullptr, destroy_green};
  CUdevResource resource{};
  CUdevResourceDesc descriptor{};
  ASSERT_EQ(get_resource(0, &resource, CU_DEV_RESOURCE_TYPE_SM), CUDA_SUCCESS);
  ASSERT_EQ(generate_desc(&descriptor, &resource, 1), CUDA_SUCCESS);
  ASSERT_EQ(create_green(&green.context, descriptor, 0, CU_GREEN_CTX_DEFAULT_STREAM), CUDA_SUCCESS);
  ASSERT_EQ(create_stream(&green.stream, green.context, CU_STREAM_NON_BLOCKING, 0), CUDA_SUCCESS);
  CUcontext context = nullptr;
  CUgreenCtx observed_green = nullptr;
  ASSERT_EQ(get_context(green.stream, &context, &observed_green), CUDA_SUCCESS);
  ASSERT_EQ(observed_green, green.context);
  ASSERT_NE(observed_green, nullptr);
  unsigned int flags = 0;
  ASSERT_EQ(cudaStreamGetFlags(green.stream, &flags), cudaSuccess);
  ASSERT_NE(flags & cudaStreamNonBlocking, 0u);

  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, streams_[0]));
  }
  ASSERT_TRUE(graph.is_captured());
  CudaCalls calls;
  for (int i = 0; i < 5; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, green.stream));
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
    EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
    EXPECT_TRUE(graph.is_captured());
  }
  EXPECT_EQ(calls.malloc_calls, 0u);
  EXPECT_TRUE(calls.retirements.empty());
  EXPECT_TRUE(calls.captures.empty());
  EXPECT_TRUE(calls.launches.empty());
  ASSERT_NO_FATAL_FAILURE(compare(plain, graph, 0, 2, 8, 9, streams_[0]));
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

TEST_F(ExecutionGraphTest, PooledScratchEnginesStayPlain) {
  const auto blob = build_scratch_blob();
  ASSERT_FALSE(blob.empty());
  BackendOption options[] = {option(kScratchOption, true), option(kGraphOption, true)};
  ASSERT_EQ(set_options({options, 2}), Error::Ok);
  LoadedGraphEngine engine;
  ASSERT_EQ(engine.load(blob), Error::Ok);
  ASSERT_TRUE(engine.handle()->claims_pooled_scratch);
  EXPECT_EQ(engine.handle()->execution_graph, nullptr);
  std::vector<float> input(8 * 32, 1.0f), output(8 * 32, -1.0f);
  CudaCalls calls;
  for (int i = 0; i < 4; ++i) {
    SizesType sizes[] = {8, 32};
    TensorImpl input_impl(ScalarType::Float, 2, sizes, input.data());
    TensorImpl output_impl(ScalarType::Float, 2, sizes, output.data());
    EValue input_value{Tensor(&input_impl)};
    EValue output_value{Tensor(&output_impl)};
    EValue* args[] = {&input_value, &output_value};
    ASSERT_EQ(engine.run({args, 2}, streams_[0]), Error::Ok);
    for (float value : output) {
      ASSERT_FLOAT_EQ(value, 1.0f / 8);
    }
  }
  EXPECT_TRUE(calls.captures.empty());
  EXPECT_TRUE(calls.launches.empty());
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
  unsigned int flags = 0;
  ASSERT_EQ(cudaStreamGetFlags(calls.captures.front(), &flags), cudaSuccess);
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
  size_t overlaps = 0;
  calls.during_capture = [&](cudaStream_t first_capture) {
    std::thread worker([&] {
      ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
      CudaCalls second_calls;
      ASSERT_EQ(second.run(inputs_[1], outputs_[1], 2, 8, streams_[0]), Error::Ok);
      ASSERT_EQ(second_calls.captures.size(), 1u);
      EXPECT_NE(second_calls.captures.front(), first_capture);
      EXPECT_NE(second_calls.captures.front(), streams_[0]);
      EXPECT_EQ(second_calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
      std::vector<float> output(16);
      ASSERT_EQ(cudaMemcpy(output.data(), outputs_[1], 64, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(output, std::vector<float>(16, 101.0f));
      ++overlaps;
    });
    worker.join();
  };
  ASSERT_NO_FATAL_FAILURE(compare(plain, first, 0, 2, 8, 2, streams_[0]));
  EXPECT_EQ(overlaps, 1u);
  EXPECT_EQ(calls.captures.size(), 1u);
  EXPECT_EQ(calls.launches, std::vector<cudaStream_t>(1, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, AllocationFailureUsesCallerBindingsAndSuppressesRetries) {
  for (size_t failed_allocation : {1u, 2u}) {
    for (bool growth : {false, true}) {
      SCOPED_TRACE(failed_allocation);
      SCOPED_TRACE(growth);
      LoadedGraphEngine plain, graph;
      ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
      if (growth) {
        for (int i = 0; i < 2; ++i) {
          ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, streams_[0]));
        }
        ASSERT_TRUE(graph.is_captured());
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
    for (int i = 0; i < 6; ++i) {
      ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i + 2, streams_[0]));
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("input_0"), inputs_[i % 2]);
      EXPECT_EQ(graph.handle()->exec_ctx->getTensorAddress("output_0"), outputs_[i % 2]);
      EXPECT_FALSE(graph.is_captured());
    }
    EXPECT_EQ(calls.captures.size(), 3u);
    EXPECT_TRUE(calls.launches.empty());
    EXPECT_EQ(calls.malloc_calls, 0u);
    EXPECT_EQ(calls.free_calls, 0u);
    EXPECT_EQ(calls.retirements, std::vector<cudaStream_t>(2, streams_[0]));
  }
  CudaCalls recovery;
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 4, 4, i + 8, streams_[0]));
  }
  EXPECT_EQ(recovery.captures.size(), 1u);
  EXPECT_EQ(recovery.launches, std::vector<cudaStream_t>(2, streams_[0]));
}

TEST_F(ExecutionGraphReplayTest, GrowthRetiresBuffersOnTheCallerStreamWithoutCudaFree) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  for (int i = 0; i < 2; ++i) {
    ASSERT_NO_FATAL_FAILURE(compare(plain, graph, i % 2, 2, 8, i, streams_[0]));
  }
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

TEST_F(ExecutionGraphReplayTest, HostBuffersStayCurrentOnReplay) {
  LoadedGraphEngine plain, graph;
  ASSERT_NO_FATAL_FAILURE(load_pair(plain, graph));
  std::vector<float> inputs[2] = {std::vector<float>(16), std::vector<float>(16)};
  std::vector<float> outputs[2] = {std::vector<float>(16), std::vector<float>(16)};
  std::vector<float> expected(16);
  for (int i = 0; i < 5; ++i) {
    const int slot = i % 2;
    for (size_t j = 0; j < 16; ++j) {
      inputs[slot][j] = static_cast<float>(i * 16 + j);
      outputs[slot][j] = -1000.0f;
    }
    ASSERT_EQ(plain.run(inputs[slot].data(), expected.data(), 2, 8, streams_[0]), Error::Ok);
    ASSERT_EQ(graph.run(inputs[slot].data(), outputs[slot].data(), 2, 8, streams_[0]), Error::Ok);
    EXPECT_EQ(graph.is_captured(), i >= 1);
    EXPECT_EQ(outputs[slot], expected);
    for (size_t j = 0; j < 16; ++j) {
      ASSERT_FLOAT_EQ(outputs[slot][j], inputs[slot][j] * 2.0f + 1.0f);
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

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
