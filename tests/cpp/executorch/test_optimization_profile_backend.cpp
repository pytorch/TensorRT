/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// OptimizationProfileGuard applied through TensorRTBackend::execute() on a real
// engine, as opposed to test_optimization_profile_selection.cpp, which covers the
// policy alone. The engine has two profiles with disjoint ranges, so which one ran
// is observable from which shapes succeed as well as from profiles.active.

#include "torch_tensorrt/executorch/EngineHandle.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using namespace ::executorch::aten;
using namespace ::executorch::runtime;
using ::executorch::extension::cuda::CallerStreamGuard;

constexpr int32_t kDecode = 0; // rows == 1, cols 1..8
constexpr int32_t kPrefill = 1; // rows 2..8, cols 1..32
constexpr size_t kMaxBytes = 8 * 32 * sizeof(float);

// y = 2x + 1, or, with softmax, a softmax over columns then one over rows: two
// passes over different axes do not fuse, so that engine needs activation scratch.
std::vector<uint8_t> build_blob(bool two_profiles, bool softmax = false) {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder ? builder->createNetworkV2(0) : nullptr);
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder ? builder->createBuilderConfig() : nullptr);
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{-1, -1});
  nvinfer1::ITensor* out = nullptr;
  if (softmax) {
    auto* over_cols = network->addSoftMax(*input);
    over_cols->setAxes(1u << 1);
    auto* over_rows = network->addSoftMax(*over_cols->getOutput(0));
    over_rows->setAxes(1u << 0);
    out = over_rows->getOutput(0);
  } else {
    static const float scale = 2.0f;
    static const float bias = 1.0f;
    auto* multiplier = network->addConstant({2, {1, 1}}, {nvinfer1::DataType::kFLOAT, &scale, 1});
    auto* addend = network->addConstant({2, {1, 1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
    auto* product = network->addElementWise(*input, *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
    out = network->addElementWise(*product->getOutput(0), *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)
              ->getOutput(0);
  }
  out->setName("output_0");
  network->markOutput(*out);
  struct Range {
    nvinfer1::Dims2 min, opt, max;
  };
  std::vector<Range> ranges = {{{2, 1}, {4, 16}, {8, 32}}};
  if (two_profiles) {
    ranges.insert(ranges.begin(), Range{{1, 1}, {1, 4}, {1, 8}});
  }
  for (const Range& r : ranges) {
    auto* profile = builder->createOptimizationProfile();
    if (!profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMIN, r.min) ||
        !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kOPT, r.opt) ||
        !profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMAX, r.max) ||
        config->addOptimizationProfile(profile) < 0) {
      return {};
    }
  }
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  const std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0})";
  const uint32_t metadata_offset = 32;
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const uint32_t engine_offset = (metadata_offset + metadata_size + 15) / 16 * 16;
  const uint64_t engine_size = plan->size();
  std::vector<uint8_t> blob(engine_offset + engine_size, 0);
  std::memcpy(blob.data(), "TR01", 4);
  std::memcpy(blob.data() + 4, &metadata_offset, sizeof(metadata_offset));
  std::memcpy(blob.data() + 8, &metadata_size, sizeof(metadata_size));
  std::memcpy(blob.data() + 12, &engine_offset, sizeof(engine_offset));
  std::memcpy(blob.data() + 16, &engine_size, sizeof(engine_size));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(blob.data() + engine_offset, plan->data(), plan->size());
  return blob;
}

std::vector<float> expected_output(const std::vector<float>& x, int rows, int cols, bool softmax) {
  std::vector<float> y(x.size());
  if (!softmax) {
    for (size_t i = 0; i < x.size(); ++i) {
      y[i] = x[i] * 2.0f + 1.0f;
    }
    return y;
  }
  for (int r = 0; r < rows; ++r) {
    double sum = 0;
    for (int c = 0; c < cols; ++c) {
      sum += std::exp(x[r * cols + c]);
    }
    for (int c = 0; c < cols; ++c) {
      y[r * cols + c] = static_cast<float>(std::exp(x[r * cols + c]) / sum);
    }
  }
  for (int c = 0; c < cols; ++c) {
    double sum = 0;
    for (int r = 0; r < rows; ++r) {
      sum += std::exp(y[r * cols + c]);
    }
    for (int r = 0; r < rows; ++r) {
      y[r * cols + c] = static_cast<float>(std::exp(y[r * cols + c]) / sum);
    }
  }
  return y;
}

Error set_scratch(bool enabled) {
  BackendOption option{};
  std::strncpy(option.key, kSharedActivationScratchKey, sizeof(option.key) - 1);
  option.value = enabled;
  TensorRTBackend backend;
  BackendOptionContext context;
  return backend.set_option(context, {&option, 1});
}

class Engine {
 public:
  ~Engine() {
    if (handle_ != nullptr) {
      backend_.destroy(handle_);
    }
    cudaFree(input_);
    cudaFree(output_);
  }

  Error load(const std::vector<uint8_t>& blob, bool softmax = false) {
    softmax_ = softmax;
    if (cudaMalloc(&input_, kMaxBytes) != cudaSuccess || cudaMalloc(&output_, kMaxBytes) != cudaSuccess) {
      return Error::MemoryAllocationFailed;
    }
    BackendInitContext context(&allocator_, nullptr, nullptr, nullptr);
    FreeableBuffer processed(blob.data(), blob.size(), nullptr);
    auto result = backend_.init(context, &processed, {});
    if (!result.ok()) {
      return result.error();
    }
    handle_ = static_cast<EngineHandle*>(result.get());
    return Error::Ok;
  }

  // Runs on device buffers. The host copies are made by the caller's thread
  // before and after, so a fresh thread can call this with no CUDA call of its own.
  Error execute(int rows, int cols, std::optional<cudaStream_t> stream = std::nullopt) {
    SizesType sizes[] = {rows, cols};
    TensorImpl in(ScalarType::Float, 2, sizes, input_);
    TensorImpl out(ScalarType::Float, 2, sizes, output_);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    if (stream) {
      CallerStreamGuard guard(*stream);
      return backend_.execute(context, handle_, {args, 2});
    }
    return backend_.execute(context, handle_, {args, 2});
  }

  void stage(int rows, int cols, float seed) {
    input_host_.resize(static_cast<size_t>(rows) * cols);
    for (size_t i = 0; i < input_host_.size(); ++i) {
      input_host_[i] = seed + static_cast<float>(i % 13) / 8.0f;
    }
    ASSERT_EQ(cudaMemcpy(input_, input_host_.data(), input_host_.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(output_, 0xff, kMaxBytes), cudaSuccess);
  }

  void check(int rows, int cols) {
    std::vector<float> actual(input_host_.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), output_, actual.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    const auto expected = expected_output(input_host_, rows, cols, softmax_);
    for (size_t i = 0; i < actual.size(); ++i) {
      ASSERT_NEAR(actual[i], expected[i], 1e-4f) << rows << "x" << cols << " element " << i;
    }
  }

  // Stages, runs, and checks the result on the calling thread.
  Error run(int rows, int cols, float seed = 0.0f, std::optional<cudaStream_t> stream = std::nullopt) {
    stage(rows, cols, seed);
    const Error err = execute(rows, cols, stream);
    if (err == Error::Ok) {
      if (stream) {
        EXPECT_EQ(cudaStreamSynchronize(*stream), cudaSuccess);
      }
      check(rows, cols);
    }
    return err;
  }

  EngineHandle* handle() const {
    return handle_;
  }

  int32_t active() const {
    return handle_->profiles.active;
  }

 private:
  TensorRTBackend backend_;
  alignas(EngineHandle) uint8_t arena_[4096]{};
  MemoryAllocator allocator_{sizeof(arena_), arena_};
  EngineHandle* handle_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
  std::vector<float> input_host_;
  bool softmax_ = false;
};

class ProfileBackendTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0 || !blob_.empty()) {
      return;
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    blob_ = build_blob(true);
    single_blob_ = build_blob(false);
    softmax_blob_ = build_blob(true, true);
  }

  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      const char* required = std::getenv("TORCHTRT_EXECUTORCH_REQUIRE_CUDA");
      if (required && std::strcmp(required, "1") == 0) {
        FAIL() << "This run requires CUDA; no profile test could execute";
      }
      GTEST_SKIP() << "No CUDA device; profile switching is not covered";
    }
    ASSERT_FALSE(blob_.empty());
    ASSERT_FALSE(single_blob_.empty());
    ASSERT_FALSE(softmax_blob_.empty());
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(set_scratch(false), Error::Ok);
  }

  void TearDown() override {
    EXPECT_EQ(set_scratch(false), Error::Ok);
  }

  static std::vector<uint8_t> blob_;
  static std::vector<uint8_t> single_blob_;
  static std::vector<uint8_t> softmax_blob_;
};

std::vector<uint8_t> ProfileBackendTest::blob_;
std::vector<uint8_t> ProfileBackendTest::single_blob_;
std::vector<uint8_t> ProfileBackendTest::softmax_blob_;

TEST_F(ProfileBackendTest, WithoutAGuardOnlyProfileZeroRuns) {
  Engine e;
  ASSERT_EQ(e.load(blob_), Error::Ok);
  EXPECT_EQ(e.run(1, 8), Error::Ok);
  EXPECT_EQ(e.run(4, 16), Error::InvalidArgument);
  EXPECT_EQ(e.active(), kDecode);
}

TEST_F(ProfileBackendTest, AutomaticSelectsTheProfileThatFitsEachCall) {
  Engine e;
  ASSERT_EQ(e.load(blob_), Error::Ok);
  auto guard = OptimizationProfileGuard::automatic();
  for (int i = 0; i < 3; ++i) {
    ASSERT_EQ(e.run(4, 16, i), Error::Ok);
    EXPECT_EQ(e.active(), kPrefill);
    ASSERT_EQ(e.run(1, 8, i), Error::Ok);
    EXPECT_EQ(e.active(), kDecode);
  }
  EXPECT_EQ(e.run(9, 8), Error::InvalidArgument) << "no profile takes 9 rows";
  EXPECT_EQ(e.active(), kDecode);
}

TEST_F(ProfileBackendTest, APinnedProfileRunsOnlyItsOwnShapes) {
  Engine e;
  ASSERT_EQ(e.load(blob_), Error::Ok);
  {
    OptimizationProfileGuard guard(kPrefill);
    ASSERT_EQ(e.run(4, 16), Error::Ok);
    EXPECT_EQ(e.active(), kPrefill);
    EXPECT_EQ(e.run(1, 8), Error::InvalidArgument) << "a pin is not a hint";
    EXPECT_EQ(e.active(), kPrefill);
  }
  ASSERT_EQ(e.run(1, 8), Error::Ok);
  EXPECT_EQ(e.active(), kDecode);
}

TEST_F(ProfileBackendTest, APinTheEngineLacksFailsWithoutSwitching) {
  Engine e;
  ASSERT_EQ(e.load(blob_), Error::Ok);
  for (int64_t index : {int64_t{2}, int64_t{-1}, int64_t{1} << 32}) {
    OptimizationProfileGuard guard(index);
    EXPECT_EQ(e.run(1, 8), Error::InvalidArgument) << index;
    EXPECT_EQ(e.active(), kDecode) << index;
  }
}

TEST_F(ProfileBackendTest, ASingleProfileEngineIgnoresAPositivePinAndRejectsANegativeOne) {
  Engine e;
  ASSERT_EQ(e.load(single_blob_), Error::Ok);
  {
    OptimizationProfileGuard guard(1);
    EXPECT_EQ(e.run(4, 16), Error::Ok);
  }
  {
    OptimizationProfileGuard guard(-1);
    EXPECT_EQ(e.run(4, 16), Error::InvalidArgument);
  }
  EXPECT_EQ(e.active(), 0);
}

TEST_F(ProfileBackendTest, NestedGuardsRestoreTheEnclosingRequest) {
  Engine e;
  ASSERT_EQ(e.load(blob_), Error::Ok);
  {
    OptimizationProfileGuard outer(kPrefill);
    {
      auto inner = OptimizationProfileGuard::automatic();
      ASSERT_EQ(e.run(1, 8), Error::Ok);
      EXPECT_EQ(e.active(), kDecode);
    }
    // Back under the pin: a restored automatic() would switch to profile 1 too,
    // so check that decode shapes are refused as well.
    ASSERT_EQ(e.run(4, 16), Error::Ok);
    EXPECT_EQ(e.active(), kPrefill);
    EXPECT_EQ(e.run(1, 8), Error::InvalidArgument);
    {
      OptimizationProfileGuard inner(kDecode);
      ASSERT_EQ(e.run(1, 8), Error::Ok);
      EXPECT_EQ(e.active(), kDecode);
    }
    ASSERT_EQ(e.run(4, 16), Error::Ok);
    EXPECT_EQ(e.active(), kPrefill);
  }
  EXPECT_EQ(e.run(4, 16), Error::InvalidArgument) << "no guard left, so profile 0";
  ASSERT_EQ(e.run(1, 8), Error::Ok);
  EXPECT_EQ(e.active(), kDecode);
}

TEST_F(ProfileBackendTest, TheRequestBelongsToTheThreadThatMadeIt) {
  Engine e;
  ASSERT_EQ(e.load(blob_), Error::Ok);
  OptimizationProfileGuard guard(kPrefill);
  e.stage(1, 8, 0.0f);
  Error result = Error::Internal;
  std::thread other([&] { result = e.execute(1, 8); });
  other.join();
  EXPECT_EQ(result, Error::Ok) << "the other thread has no guard, so it runs profile 0";
  EXPECT_EQ(e.active(), kDecode);
  ASSERT_NO_FATAL_FAILURE(e.check(1, 8));
}

// No CUDA call runs on the worker before execute(), so it starts without a
// current context, and the switch is the first thing to need one.
TEST_F(ProfileBackendTest, AFreshThreadCanSwitchProfiles) {
  cudaStream_t caller = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&caller, cudaStreamNonBlocking), cudaSuccess);
  for (int mode = 0; mode < 3; ++mode) {
    SCOPED_TRACE(mode == 0 ? "pinned" : mode == 1 ? "automatic" : "pinned on a caller stream");
    Engine e;
    ASSERT_EQ(e.load(blob_), Error::Ok);
    e.stage(4, 16, 1.0f);
    Error result = Error::Internal;
    std::thread fresh([&] {
      if (mode == 1) {
        auto guard = OptimizationProfileGuard::automatic();
        result = e.execute(4, 16);
      } else {
        OptimizationProfileGuard guard(kPrefill);
        result = mode == 2 ? e.execute(4, 16, caller) : e.execute(4, 16);
      }
    });
    fresh.join();
    EXPECT_EQ(result, Error::Ok);
    if (result != Error::Ok) {
      continue;
    }
    EXPECT_EQ(e.active(), kPrefill);
    ASSERT_NO_FATAL_FAILURE(e.check(4, 16));
  }
  EXPECT_EQ(cudaStreamDestroy(caller), cudaSuccess);
}

// Handles of one shared engine each have their own context, so each keeps its
// own active profile, including while both run at once.
TEST_F(ProfileBackendTest, HandlesOfASharedEngineSwitchIndependently) {
  Engine prefill, decode;
  ASSERT_EQ(prefill.load(blob_), Error::Ok);
  ASSERT_EQ(decode.load(blob_), Error::Ok);
  ASSERT_EQ(prefill.handle()->engine.get(), decode.handle()->engine.get()) << "the engine was not shared";
  cudaStream_t streams[2]{};
  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
  }
  auto loop = [&](Engine& e, int32_t profile, int rows, int cols, cudaStream_t stream) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    OptimizationProfileGuard guard(profile);
    for (int i = 0; i < 50; ++i) {
      ASSERT_EQ(e.run(rows, cols, i, stream), Error::Ok) << "iteration " << i;
      ASSERT_EQ(e.active(), profile);
    }
  };
  std::thread a([&] { loop(prefill, kPrefill, 4, 16, streams[0]); });
  std::thread b([&] { loop(decode, kDecode, 1, 8, streams[1]); });
  a.join();
  b.join();
  for (auto& s : streams) {
    EXPECT_EQ(cudaStreamDestroy(s), cudaSuccess);
  }
}

// A switch is submitted before the pooled scratch is claimed. That is safe only
// because a switch writes none of a user-managed context's scratch; if it did,
// one handle's switch would land in a buffer the other is using.
TEST_F(ProfileBackendTest, PooledScratchHandlesSwitchProfilesConcurrently) {
  ASSERT_EQ(set_scratch(true), Error::Ok);
  Engine prefill, decode;
  ASSERT_EQ(prefill.load(softmax_blob_, true), Error::Ok);
  ASSERT_EQ(decode.load(softmax_blob_, true), Error::Ok);
  ASSERT_TRUE(prefill.handle()->claims_pooled_scratch) << "the engine needs no scratch, so the pool is not exercised";
  ASSERT_TRUE(decode.handle()->claims_pooled_scratch);
  cudaStream_t streams[2]{};
  for (auto& s : streams) {
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
  }
  // Each thread alternates profiles, so switches interleave with the other handle's work.
  auto loop = [&](Engine& e, int phase, cudaStream_t stream) {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto guard = OptimizationProfileGuard::automatic();
    for (int i = 0; i < 40; ++i) {
      const bool big = (i + phase) % 2 == 0;
      ASSERT_EQ(e.run(big ? 8 : 1, big ? 32 : 8, i % 5, stream), Error::Ok) << "iteration " << i;
      ASSERT_EQ(e.active(), big ? kPrefill : kDecode);
    }
  };
  std::thread a([&] { loop(prefill, 0, streams[0]); });
  std::thread b([&] { loop(decode, 1, streams[1]); });
  a.join();
  b.join();
  for (auto& s : streams) {
    EXPECT_EQ(cudaStreamDestroy(s), cudaSuccess);
  }
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
