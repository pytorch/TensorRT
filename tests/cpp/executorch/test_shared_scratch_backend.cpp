/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Exercises the shared activation-scratch pool through the delegate that uses
// it: the runtime option that turns it on, the per-engine capture of that
// option, and the pooled execute() path -- the kUSER_MANAGED context, the
// updateDeviceMemorySizeForShapes/setDeviceMemoryV2 pair, the enqueue handoff
// between two caller streams, the growth a larger engine forces on a pool a
// smaller one already allocated, and two threads submitting against one pooled
// buffer at once.
//
// The TensorRT engine is built here rather than loaded from a .pte so the target
// carries no exported artifact, at the cost of a few seconds of builder time.
//
// COVERAGE LIMIT: every test below needs a CUDA device and a TensorRT that can
// build an engine. Without one the whole suite skips and covers nothing, so a
// green run on a host with no GPU says nothing about the pool.

#include "torch_tensorrt/executorch/TensorRTBackend.h"
#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/freeable_buffer.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/platform/runtime.h>

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::BackendOption;
using ::executorch::runtime::BackendOptionContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::Span;

// Spelled out rather than taken from SharedScratchPool.h: a test that reads the
// key through the production constant cannot pin the key's value.
constexpr char kOptionKey[] = "use_shared_activation_scratch";

constexpr int kRows = 2048;
constexpr int kCols = 2048;
constexpr std::size_t kElems = static_cast<std::size_t>(kRows) * static_cast<std::size_t>(kCols);
constexpr std::size_t kBytes = kElems * sizeof(float);

// A second scratch-needing engine, four times the elements of the one above, so
// loading it after that one drives the pool's growth path. Every other engine in
// this file asks for the same size, which is why nothing else reaches it.
constexpr int kBigRows = 4096;
constexpr int kBigCols = 4096;

// Engines loaded together in the memory test. Four is enough for the private
// case to cost 4x the scratch and the pooled case 1x.
constexpr int kEngineCount = 4;

// A value neither network below can produce, so an output comparison cannot be
// satisfied by an execute() that never reached the engine.
constexpr float kSentinel = -7.0f;

// Below this the memory comparison cannot see past allocator granularity, so the
// test reports that its network stopped producing measurable scratch instead of
// passing on a difference it cannot resolve.
constexpr std::size_t kMinMeasurableScratch = 4u << 20;

// ---------------------------------------------------------------------------
// A TensorRT engine, built here, wrapped in the delegate's blob wire format
// ---------------------------------------------------------------------------

constexpr char kMagic[4] = {'T', 'R', '0', '1'};
constexpr std::uint32_t kMetadataOffsetField = 4;
constexpr std::uint32_t kMetadataSizeField = 8;
constexpr std::uint32_t kEngineOffsetField = 12;
constexpr std::uint32_t kEngineSizeField = 16;
constexpr std::uint32_t kHeaderSize = 32;
constexpr std::uint32_t kEngineAlignment = 16;

class BuilderLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
  }
};

template <typename T>
void write_field(std::vector<std::uint8_t>& blob, std::size_t offset, T value) {
  std::memcpy(blob.data() + offset, &value, sizeof(value));
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

// Two softmaxes over different axes sit between the pointwise layers so the
// chain cannot collapse into a single pass, which is what keeps the engine's
// activation requirement large enough for the memory comparison to resolve.
bool add_scratch_needing_net(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input) {
  static const float kAddend = 0.125f;
  static const float kScale = 1.5f;

  nvinfer1::IConstantLayer* addend =
      network.addConstant(nvinfer1::Dims3{1, 1, 1}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &kAddend, 1});
  nvinfer1::IConstantLayer* scale =
      network.addConstant(nvinfer1::Dims3{1, 1, 1}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &kScale, 1});
  if (addend == nullptr || scale == nullptr) {
    return false;
  }

  nvinfer1::IElementWiseLayer* shifted =
      network.addElementWise(input, *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
  nvinfer1::ISoftMaxLayer* over_cols = network.addSoftMax(*shifted->getOutput(0));
  over_cols->setAxes(1u << 2);
  nvinfer1::ISoftMaxLayer* over_rows = network.addSoftMax(*over_cols->getOutput(0));
  over_rows->setAxes(1u << 1);
  nvinfer1::IElementWiseLayer* scaled =
      network.addElementWise(*over_rows->getOutput(0), *scale->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
  scaled->getOutput(0)->setName("output_0");
  network.markOutput(*scaled->getOutput(0));
  return true;
}

// TensorRT routes a pointwise chain through the I/O tensors alone, so this
// engine's activation requirement is zero -- the same answer it gives for a
// failed query. Every layer is parameterless, because a default alpha or beta
// can collapse a chain to a constant and make an output comparison vacuous.
bool add_scratch_free_net(nvinfer1::INetworkDefinition& network, nvinfer1::ITensor& input) {
  static const nvinfer1::ActivationType kChain[] = {
      nvinfer1::ActivationType::kSIGMOID,
      nvinfer1::ActivationType::kTANH,
      nvinfer1::ActivationType::kSOFTSIGN,
      nvinfer1::ActivationType::kSIGMOID,
      nvinfer1::ActivationType::kTANH,
      nvinfer1::ActivationType::kSOFTSIGN,
  };
  nvinfer1::ITensor* t = &input;
  for (const nvinfer1::ActivationType op : kChain) {
    nvinfer1::IActivationLayer* layer = network.addActivation(*t, op);
    if (layer == nullptr) {
      return false;
    }
    t = layer->getOutput(0);
  }
  t->setName("output_0");
  network.markOutput(*t);
  return true;
}

std::vector<std::uint8_t> build_engine_blob(bool needs_scratch, int rows = kRows, int cols = kCols) {
  static BuilderLogger logger;

  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (builder == nullptr) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  if (network == nullptr) {
    return {};
  }

  nvinfer1::ITensor* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{1, rows, cols});
  if (input == nullptr) {
    return {};
  }
  const bool built = needs_scratch ? add_scratch_needing_net(*network, *input) : add_scratch_free_net(*network, *input);
  if (!built) {
    return {};
  }

  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (config == nullptr) {
    return {};
  }
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  const nvinfer1::Dims3 shape{1, rows, cols};
  profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMIN, shape);
  profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kOPT, shape);
  profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMAX, shape);
  config->addOptimizationProfile(profile);

  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (plan == nullptr) {
    return {};
  }

  const std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],)"
      R"("hardware_compatible":false,"device_id":0})";
  const auto metadata_offset = static_cast<std::uint32_t>(kHeaderSize);
  const auto metadata_size = static_cast<std::uint32_t>(metadata.size());
  const auto engine_offset = static_cast<std::uint32_t>(align_up(metadata_offset + metadata_size, kEngineAlignment));

  std::vector<std::uint8_t> blob(static_cast<std::size_t>(engine_offset) + plan->size(), 0);
  std::memcpy(blob.data(), kMagic, sizeof(kMagic));
  write_field(blob, kMetadataOffsetField, metadata_offset);
  write_field(blob, kMetadataSizeField, metadata_size);
  write_field(blob, kEngineOffsetField, engine_offset);
  write_field(blob, kEngineSizeField, static_cast<std::uint64_t>(plan->size()));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(blob.data() + engine_offset, plan->data(), plan->size());
  return blob;
}

// The activation scratch one context of the shared engine needs, read the way
// execute() reads it. Zero if the engine could not be measured.
std::size_t measure_engine_scratch(const std::vector<std::uint8_t>& blob, int rows = kRows, int cols = kCols) {
  static BuilderLogger logger;
  TensorRTBlobHeader header;
  if (!TensorRTBlobHeader::parse(blob.data(), blob.size(), header)) {
    return 0;
  }
  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  if (runtime == nullptr) {
    return 0;
  }
  TRTUniquePtr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(TensorRTBlobHeader::engine_data(blob.data(), header), header.engine_size));
  if (engine == nullptr) {
    return 0;
  }
  // kUSER_MANAGED so the probe context itself allocates no scratch to measure.
  TRTUniquePtr<nvinfer1::IExecutionContext> ctx(
      engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (ctx == nullptr) {
    return 0;
  }
  if (!ctx->setInputShape("input_0", nvinfer1::Dims3{1, rows, cols})) {
    return 0;
  }
  return ctx->updateDeviceMemorySizeForShapes();
}

// What the engine reports it needs, read the way init() reads it. A negative
// result means the blob could not be opened, which no engine reports and which
// no test may mistake for a scratch-free engine.
std::int64_t engine_scratch_requirement(const std::vector<std::uint8_t>& blob) {
  static BuilderLogger logger;
  TensorRTBlobHeader header;
  if (!TensorRTBlobHeader::parse(blob.data(), blob.size(), header)) {
    return -1;
  }
  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  if (runtime == nullptr) {
    return -1;
  }
  TRTUniquePtr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(TensorRTBlobHeader::engine_data(blob.data(), header), header.engine_size));
  if (engine == nullptr) {
    return -1;
  }
  return engine->getDeviceMemorySizeV2();
}

// ---------------------------------------------------------------------------
// One loaded delegate handle plus the device-resident I/O its execute() needs
// ---------------------------------------------------------------------------

// Reproducible on both sides and non-uniform: a constant input would make the
// softmaxes uniform and stop the output depending on the tensor under test.
float pattern(std::size_t index, std::uint32_t seed) {
  std::uint32_t h = static_cast<std::uint32_t>(index) * 2654435761u + seed * 40503u;
  h ^= h >> 15;
  return static_cast<float>(h % 1000u) / 500.0f - 1.0f;
}

class LoadedEngine {
 public:
  LoadedEngine() = default;
  LoadedEngine(const LoadedEngine&) = delete;
  LoadedEngine& operator=(const LoadedEngine&) = delete;

  ~LoadedEngine() {
    if (handle_ != nullptr) {
      backend_.destroy(handle_);
    }
    cudaFree(device_in_);
    cudaFree(device_out_);
  }

  // Loads the blob through the backend, capturing whatever the shared-scratch
  // option is set to at this moment. `rows`/`cols` must be the shape the blob was
  // built for.
  Error load(const std::vector<std::uint8_t>& blob, std::uint32_t seed, int rows = kRows, int cols = kCols) {
    rows_ = static_cast<SizesType>(rows);
    cols_ = static_cast<SizesType>(cols);
    std::vector<float> host_in(elems());
    for (std::size_t i = 0; i < elems(); ++i) {
      host_in[i] = pattern(i, seed);
    }
    if (cudaMalloc(&device_in_, bytes()) != cudaSuccess || cudaMalloc(&device_out_, bytes()) != cudaSuccess) {
      return Error::MemoryAllocationFailed;
    }
    if (cudaMemcpy(device_in_, host_in.data(), bytes(), cudaMemcpyHostToDevice) != cudaSuccess) {
      return Error::Internal;
    }

    arena_storage_.resize(kArenaBytes);
    arena_ = std::make_unique<MemoryAllocator>(static_cast<std::uint32_t>(kArenaBytes), arena_storage_.data());
    BackendInitContext init_context(arena_.get());
    FreeableBuffer processed(blob.data(), blob.size(), nullptr);
    const auto result = backend_.init(init_context, &processed, ArrayRef<CompileSpec>{});
    if (!result.ok()) {
      return result.error();
    }
    handle_ = result.get();
    return Error::Ok;
  }

  bool fill_output(float value) {
    const std::vector<float> host(elems(), value);
    return cudaMemcpy(device_out_, host.data(), bytes(), cudaMemcpyHostToDevice) == cudaSuccess;
  }

  // Runs one inference on `stream`. Returns without waiting for the enqueue,
  // which is the state the pool's handoff exists to order.
  Error run(cudaStream_t stream) {
    // Separate arrays: execute() resizes the output tensor to the shape TensorRT
    // inferred, which writes through whichever array that tensor was given.
    SizesType in_sizes[3] = {1, rows_, cols_};
    SizesType out_sizes[3] = {1, rows_, cols_};
    ::executorch::aten::TensorImpl in_impl(ScalarType::Float, 3, in_sizes, device_in_);
    ::executorch::aten::TensorImpl out_impl(ScalarType::Float, 3, out_sizes, device_out_);
    ::executorch::aten::Tensor in_tensor(&in_impl);
    ::executorch::aten::Tensor out_tensor(&out_impl);
    EValue in_value(in_tensor);
    EValue out_value(out_tensor);
    EValue* args[2] = {&in_value, &out_value};

    BackendExecutionContext exec_context;
    ::executorch::extension::cuda::CallerStreamGuard guard(stream);
    return backend_.execute(exec_context, handle_, Span<EValue*>(args, 2));
  }

  std::vector<float> read_output() const {
    std::vector<float> host_out(elems());
    if (cudaMemcpy(host_out.data(), device_out_, bytes(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      host_out.clear();
    }
    return host_out;
  }

  const EngineHandle* handle() const {
    return static_cast<const EngineHandle*>(handle_);
  }

  std::size_t elems() const {
    return static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
  }

  std::size_t bytes() const {
    return elems() * sizeof(float);
  }

 private:
  // EngineHandle is placement-newed into this arena by init(), and the arena is
  // never reset, so it only has to hold one instance.
  static constexpr std::size_t kArenaBytes = 4096;

  TensorRTBackend backend_;
  std::vector<std::uint8_t> arena_storage_;
  std::unique_ptr<MemoryAllocator> arena_;
  DelegateHandle* handle_ = nullptr;
  void* device_in_ = nullptr;
  void* device_out_ = nullptr;
  SizesType rows_ = kRows;
  SizesType cols_ = kCols;
};

std::size_t device_bytes_in_use() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return 0;
  }
  return total_bytes - free_bytes;
}

Error set_shared_scratch(TensorRTBackend& backend, bool enabled) {
  BackendOption option;
  std::strncpy(option.key, kOptionKey, sizeof(option.key) - 1);
  option.value = enabled;
  BackendOption options[1] = {option};
  BackendOptionContext context;
  return backend.set_option(context, Span<BackendOption>(options, 1));
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class SharedScratchBackendTest : public ::testing::Test {
 protected:
  // Building the engine dominates the runtime of this target, so it is built
  // once and every test loads the same blob.
  static void SetUpTestSuite() {
    ::executorch::runtime::runtime_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      return;
    }
    blob_ = build_engine_blob(true);
    scratch_free_blob_ = build_engine_blob(false);
    big_blob_ = build_engine_blob(true, kBigRows, kBigCols);
    if (blob_.empty() || scratch_free_blob_.empty() || big_blob_.empty()) {
      return;
    }
    scratch_bytes_ = measure_engine_scratch(blob_);
    big_scratch_bytes_ = measure_engine_scratch(big_blob_, kBigRows, kBigCols);
    engine_bytes_ = engine_scratch_requirement(blob_);
    scratch_free_engine_bytes_ = engine_scratch_requirement(scratch_free_blob_);
  }

  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "no CUDA device: the shared-scratch backend path is not covered by this run";
    }
    ASSERT_FALSE(blob_.empty()) << "TensorRT could not build the fixture engine";
    ASSERT_FALSE(scratch_free_blob_.empty()) << "TensorRT could not build the scratch-free fixture engine";
    ASSERT_FALSE(big_blob_.empty()) << "TensorRT could not build the larger fixture engine";
    ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);
  }

  void TearDown() override {
    set_shared_scratch(backend_, false);
  }

  const std::vector<std::uint8_t>& blob() const {
    return blob_;
  }

  const std::vector<std::uint8_t>& scratch_free_blob() const {
    return scratch_free_blob_;
  }

  const std::vector<std::uint8_t>& big_blob() const {
    return big_blob_;
  }

  TensorRTBackend backend_;
  static std::vector<std::uint8_t> blob_;
  static std::vector<std::uint8_t> scratch_free_blob_;
  static std::vector<std::uint8_t> big_blob_;
  static std::size_t scratch_bytes_;
  static std::size_t big_scratch_bytes_;
  static std::int64_t engine_bytes_;
  static std::int64_t scratch_free_engine_bytes_;
};

std::vector<std::uint8_t> SharedScratchBackendTest::blob_;
std::vector<std::uint8_t> SharedScratchBackendTest::scratch_free_blob_;
std::vector<std::uint8_t> SharedScratchBackendTest::big_blob_;
std::size_t SharedScratchBackendTest::scratch_bytes_ = 0;
std::size_t SharedScratchBackendTest::big_scratch_bytes_ = 0;
std::int64_t SharedScratchBackendTest::engine_bytes_ = -1;
std::int64_t SharedScratchBackendTest::scratch_free_engine_bytes_ = -1;

// ---------------------------------------------------------------------------
// set_option
// ---------------------------------------------------------------------------

// The foreign key is sent from both settings, because from one of them the test
// cannot tell a key that is ignored from a key that resets the setting to that
// value.
TEST_F(SharedScratchBackendTest, SetOptionAcceptsAKeyThisBackendDoesNotRead) {
  BackendOption foreign;
  std::strncpy(foreign.key, "some_other_backends_option", sizeof(foreign.key) - 1);
  foreign.value = 7;
  BackendOption options[1] = {foreign};
  BackendOptionContext context;

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  EXPECT_EQ(backend_.set_option(context, Span<BackendOption>(options, 1)), Error::Ok);
  LoadedEngine after_on;
  ASSERT_EQ(after_on.load(blob(), 1), Error::Ok);
  EXPECT_TRUE(after_on.handle()->shared_scratch) << "a foreign key turned the shared-scratch setting off";

  ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);
  EXPECT_EQ(backend_.set_option(context, Span<BackendOption>(options, 1)), Error::Ok);
  LoadedEngine after_off;
  ASSERT_EQ(after_off.load(blob(), 12), Error::Ok);
  EXPECT_FALSE(after_off.handle()->shared_scratch) << "a foreign key turned the shared-scratch setting on";
}

TEST_F(SharedScratchBackendTest, SetOptionStoresTheBooleanItIsGiven) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 2), Error::Ok);
  EXPECT_TRUE(pooled.handle()->shared_scratch);

  ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);
  LoadedEngine priv;
  ASSERT_EQ(priv.load(blob(), 3), Error::Ok);
  EXPECT_FALSE(priv.handle()->shared_scratch);
}

TEST_F(SharedScratchBackendTest, SetOptionRejectsANonBooleanAndLeavesTheSettingAlone) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  BackendOption wrong_type;
  std::strncpy(wrong_type.key, kOptionKey, sizeof(wrong_type.key) - 1);
  // The int has to coerce to the opposite of the setting above: one that coerced
  // to the same value would leave the setting exactly where the assertion at the
  // end expects to find it, whether it was rejected or not.
  wrong_type.value = 0;
  BackendOption options[1] = {wrong_type};
  BackendOptionContext context;
  EXPECT_EQ(backend_.set_option(context, Span<BackendOption>(options, 1)), Error::InvalidArgument);

  LoadedEngine engine;
  ASSERT_EQ(engine.load(blob(), 4), Error::Ok);
  EXPECT_TRUE(engine.handle()->shared_scratch) << "a rejected option still moved the shared-scratch setting";
}

// A context's allocation strategy is fixed when the context is created, so the
// option cannot be re-read per call.
TEST_F(SharedScratchBackendTest, EachEngineCapturesTheSettingInEffectAtItsOwnLoad) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 5), Error::Ok);

  ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);
  LoadedEngine priv;
  ASSERT_EQ(priv.load(blob(), 6), Error::Ok);

  EXPECT_TRUE(pooled.handle()->shared_scratch);
  EXPECT_FALSE(priv.handle()->shared_scratch);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  EXPECT_EQ(pooled.run(stream), Error::Ok);
  EXPECT_EQ(priv.run(stream), Error::Ok);
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// ---------------------------------------------------------------------------
// The pooled execute() path
// ---------------------------------------------------------------------------

TEST_F(SharedScratchBackendTest, APooledEngineProducesWhatAPrivateScratchEngineProduces) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  LoadedEngine priv;
  ASSERT_EQ(priv.load(blob(), 7), Error::Ok);
  // Two arms on the same setting produce the same bytes whichever setting that
  // is, so the comparison at the end is worth nothing unless each arm is pinned
  // to the side it stands for.
  ASSERT_FALSE(priv.handle()->shared_scratch);
  ASSERT_EQ(priv.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> expected = priv.read_output();

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 7), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  ASSERT_EQ(pooled.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> actual = pooled.read_output();

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_EQ(expected.size(), kElems);
  ASSERT_EQ(actual.size(), kElems);
  // A degenerate output would make the comparison above pass without depending
  // on the engine having run.
  bool varies = false;
  for (std::size_t i = 1; i < kElems && !varies; ++i) {
    varies = expected[i] != expected[0];
  }
  EXPECT_TRUE(varies) << "the reference output is constant, so the comparison proves nothing";
  EXPECT_EQ(std::memcmp(expected.data(), actual.data(), kBytes), 0);
}

TEST_F(SharedScratchBackendTest, PooledEnginesShareOneActivationScratchAllocation) {
  ASSERT_GE(scratch_bytes_, kMinMeasurableScratch)
      << "the fixture engine reports " << scratch_bytes_
      << " bytes of activation scratch, too little for the memory comparison to resolve";

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // One load and run first, so the one-time TensorRT runtime and CUDA module
  // allocations land outside both measurements.
  {
    LoadedEngine warmup;
    ASSERT_EQ(warmup.load(blob(), 8), Error::Ok);
    ASSERT_EQ(warmup.run(stream), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  }

  std::size_t private_cost = 0;
  {
    const std::size_t before = device_bytes_in_use();
    std::vector<std::unique_ptr<LoadedEngine>> engines;
    for (int i = 0; i < kEngineCount; ++i) {
      engines.push_back(std::make_unique<LoadedEngine>());
      ASSERT_EQ(engines.back()->load(blob(), 9), Error::Ok);
      ASSERT_EQ(engines.back()->run(stream), Error::Ok);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const std::size_t after = device_bytes_in_use();
    // The subtraction is unsigned, so a fall in device-wide usage would wrap it
    // to a number that satisfies the comparison at the end for free.
    ASSERT_GE(after, before) << "device-wide memory in use fell across the private-scratch measurement, so "
                                "something outside this test is releasing memory on this device";
    private_cost = after - before;
  }

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  std::size_t pooled_cost = 0;
  {
    const std::size_t before = device_bytes_in_use();
    std::vector<std::unique_ptr<LoadedEngine>> engines;
    for (int i = 0; i < kEngineCount; ++i) {
      engines.push_back(std::make_unique<LoadedEngine>());
      ASSERT_EQ(engines.back()->load(blob(), 9), Error::Ok);
      ASSERT_EQ(engines.back()->run(stream), Error::Ok);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const std::size_t after = device_bytes_in_use();
    ASSERT_GE(after, before) << "device-wide memory in use fell across the pooled measurement, so "
                                "something outside this test is releasing memory on this device";
    pooled_cost = after - before;
  }

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  // Half the ideal saving, which leaves room for allocator granularity without
  // admitting a run in which every context still carries its own scratch.
  const std::size_t expected_saving = (kEngineCount - 1) * scratch_bytes_ / 2;
  EXPECT_GE(private_cost, pooled_cost + expected_saving)
      << kEngineCount << " engines cost " << private_cost << " bytes with private scratch and " << pooled_cost
      << " pooled, against " << scratch_bytes_ << " bytes of scratch each";
}

// ---------------------------------------------------------------------------
// Growing the pool
// ---------------------------------------------------------------------------

// Runs a four-times-larger engine after a smaller one to reach the growth path,
// which nothing else in this file does.
//
// The bounds cover the second allocation and the free of the buffer it replaces,
// not the host wait before that free: cudaFree synchronizes device-wide anyway,
// so deleting the wait leaves this test green. The wait stays as the explicit
// guarantee rather than a reliance on cudaFree's implicit one.
//
// The lower bound also fails if an earlier test left the pool already large
// enough, which is how this test could otherwise pass vacuously.
TEST_F(SharedScratchBackendTest, ALargerEngineGrowsThePoolAndFreesTheBufferItReplaces) {
  ASSERT_GE(big_scratch_bytes_, scratch_bytes_ + kMinMeasurableScratch)
      << "the two fixture engines ask for " << scratch_bytes_ << " and " << big_scratch_bytes_
      << " bytes of activation scratch, too close for the growth to be measurable";

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // Private-scratch references for both engines, and the run that pays the larger
  // engine's one-time TensorRT and CUDA module costs so they land outside the
  // measurement below.
  std::vector<float> small_expected;
  std::vector<float> big_expected;
  {
    LoadedEngine small_priv;
    LoadedEngine big_priv;
    ASSERT_EQ(small_priv.load(blob(), 15), Error::Ok);
    ASSERT_EQ(big_priv.load(big_blob(), 16, kBigRows, kBigCols), Error::Ok);
    ASSERT_FALSE(small_priv.handle()->shared_scratch);
    ASSERT_FALSE(big_priv.handle()->shared_scratch);
    ASSERT_EQ(small_priv.run(stream), Error::Ok);
    ASSERT_EQ(big_priv.run(stream), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    small_expected = small_priv.read_output();
    big_expected = big_priv.read_output();
  }

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine small;
  ASSERT_EQ(small.load(blob(), 15), Error::Ok);
  ASSERT_TRUE(small.handle()->shared_scratch);
  ASSERT_EQ(small.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Loaded before the measurement starts: its weights and its I/O are not part of
  // what the growth costs.
  LoadedEngine big;
  ASSERT_EQ(big.load(big_blob(), 16, kBigRows, kBigCols), Error::Ok);
  ASSERT_TRUE(big.handle()->shared_scratch);

  const std::size_t before = device_bytes_in_use();
  ASSERT_EQ(big.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::size_t after = device_bytes_in_use();
  ASSERT_GE(after, before) << "device-wide memory in use fell across the growth, so something outside this test is "
                              "releasing memory on this device";
  const std::size_t growth_cost = after - before;
  const std::size_t difference = big_scratch_bytes_ - scratch_bytes_;

  // The pool must still serve the smaller engine after the growth moved the
  // buffer: its context holds the address it was given on its previous call, and
  // that address has been freed.
  ASSERT_TRUE(small.fill_output(kSentinel));
  ASSERT_EQ(small.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> small_actual = small.read_output();
  const std::vector<float> big_actual = big.read_output();

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  EXPECT_GE(growth_cost, difference / 2) << "the larger engine cost " << growth_cost << " bytes against a "
                                         << difference
                                         << "-byte difference in requirement, so the pool did not grow for it";
  EXPECT_LE(growth_cost, difference + scratch_bytes_ / 2)
      << "the larger engine cost " << growth_cost << " bytes, about the whole " << big_scratch_bytes_
      << "-byte buffer rather than the " << difference << "-byte difference, so the buffer it replaced was not freed";

  ASSERT_EQ(big_expected.size(), big_actual.size());
  ASSERT_FALSE(big_expected.empty());
  EXPECT_EQ(std::memcmp(big_expected.data(), big_actual.data(), big_expected.size() * sizeof(float)), 0)
      << "the engine that grew the pool did not produce what it produces with its own scratch";

  ASSERT_EQ(small_expected.size(), kElems);
  ASSERT_EQ(small_actual.size(), kElems);
  EXPECT_NE(small_expected[0], kSentinel) << "the reference output is the sentinel, so a skipped enqueue would pass";
  EXPECT_EQ(std::memcmp(small_expected.data(), small_actual.data(), kBytes), 0)
      << "the smaller engine stopped producing its own output once the growth moved the shared buffer";
}

// ---------------------------------------------------------------------------
// An engine that needs no activation scratch
// ---------------------------------------------------------------------------

// updateDeviceMemorySizeForShapes() answers a failed query and an engine that
// needs nothing identically, so execute() separates them on the engine's own
// requirement. Everything below rests on that requirement telling the two
// fixture networks apart, which is why it is asserted on its own first.
TEST_F(SharedScratchBackendTest, TheEngineLevelRequirementSeparatesTheTwoFixtureEngines) {
  EXPECT_EQ(scratch_free_engine_bytes_, 0)
      << "the pointwise chain reports " << scratch_free_engine_bytes_
      << " bytes of activation scratch, so it no longer covers the scratch-free case";
  EXPECT_GT(engine_bytes_, 0) << "the two-softmax network reports no activation scratch, so it no longer covers the "
                                 "case a failed query has to be told apart from";
}

TEST_F(SharedScratchBackendTest, EachEngineRecordsItsOwnActivationScratchRequirement) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine needing;
  LoadedEngine scratch_free;
  ASSERT_EQ(needing.load(blob(), 12), Error::Ok);
  ASSERT_EQ(scratch_free.load(scratch_free_blob(), 13), Error::Ok);

  EXPECT_EQ(static_cast<std::int64_t>(needing.handle()->engine_scratch_bytes), engine_bytes_);
  EXPECT_EQ(scratch_free.handle()->engine_scratch_bytes, 0u);
}

// Turning the pool on must not turn an engine that legitimately needs no
// activation scratch into a failure.
TEST_F(SharedScratchBackendTest, AnEngineNeedingNoActivationScratchRunsWithThePoolEnabled) {
  ASSERT_EQ(scratch_free_engine_bytes_, 0) << "the fixture engine needs scratch, so this test covers nothing";

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  LoadedEngine priv;
  ASSERT_EQ(priv.load(scratch_free_blob(), 14), Error::Ok);
  ASSERT_TRUE(priv.fill_output(kSentinel));
  ASSERT_EQ(priv.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> expected = priv.read_output();

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(scratch_free_blob(), 14), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  ASSERT_TRUE(pooled.fill_output(kSentinel));
  EXPECT_EQ(pooled.run(stream), Error::Ok) << "the pool rejected an engine that needs no activation scratch";
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> actual = pooled.read_output();

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_EQ(expected.size(), kElems);
  ASSERT_EQ(actual.size(), kElems);
  // Without these two the comparison would be satisfied by an execute() that
  // wrote nothing, and by a network whose output does not depend on its input.
  EXPECT_NE(expected[0], kSentinel) << "the reference output is the sentinel, so a skipped enqueue would pass";
  bool varies = false;
  for (std::size_t i = 1; i < kElems && !varies; ++i) {
    varies = expected[i] != expected[0];
  }
  EXPECT_TRUE(varies) << "the reference output is constant, so the comparison proves nothing";
  EXPECT_EQ(std::memcmp(expected.data(), actual.data(), kBytes), 0);
}

// ---------------------------------------------------------------------------
// The enqueue handoff, single-threaded, two caller streams
// ---------------------------------------------------------------------------

struct StreamGate {
  std::mutex mu;
  std::condition_variable cv;
  bool open = false;
  // Set when the watchdog, not the test, had to open the gate.
  std::atomic<bool> forced_open{false};
};

void CUDART_CB hold_stream(void* user_data) {
  StreamGate* gate = static_cast<StreamGate*>(user_data);
  std::unique_lock<std::mutex> lock(gate->mu);
  gate->cv.wait(lock, [gate] { return gate->open; });
}

// Long enough that the wait the test performs while the gate is shut, and the
// two enqueues before it, are nowhere near it.
constexpr std::chrono::seconds kGateWatchdog{60};

// Opens the gate and waits for the held work to drain, by two routes because two
// different things can go wrong. A held stream outlives any assertion that
// returns early, and every teardown path below -- cudaFree, the delegate
// destructor -- blocks on it, so the destructor opens the gate for a test that
// does not reach its end. That is no help if a delegate call blocks on the held
// stream instead of returning, since the calling thread then never runs the
// destructor either: the watchdog covers that, and records that it had to, so
// the outcome is a failure naming the cause rather than a process that never
// exits.
class GateRelease {
 public:
  GateRelease(StreamGate& gate, cudaStream_t stream)
      : gate_(gate), stream_(stream), deadline_(std::chrono::steady_clock::now() + kGateWatchdog) {
    watchdog_ = std::thread([this] {
      std::unique_lock<std::mutex> lock(gate_.mu);
      if (!gate_.cv.wait_until(lock, deadline_, [this] { return gate_.open; })) {
        gate_.open = true;
        gate_.forced_open.store(true);
        lock.unlock();
        gate_.cv.notify_all();
      }
    });
  }

  ~GateRelease() {
    release();
    watchdog_.join();
  }

  void release() {
    if (released_) {
      return;
    }
    released_ = true;
    {
      std::lock_guard<std::mutex> lock(gate_.mu);
      gate_.open = true;
    }
    gate_.cv.notify_all();
    cudaStreamSynchronize(stream_);
  }

 private:
  StreamGate& gate_;
  cudaStream_t stream_;
  std::chrono::steady_clock::time_point deadline_;
  std::thread watchdog_;
  bool released_ = false;
};

// Two engines on one device share one scratch buffer, so the second engine's
// enqueue must not start before the first one's has finished with it. The two
// run on different streams, which is what the README permits and what the event
// handoff is for: nothing but the handoff orders them.
TEST_F(SharedScratchBackendTest, ASecondPooledEnqueueWaitsForTheFirstOnAnotherStream) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t first_stream = nullptr;
  cudaStream_t second_stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine first;
  LoadedEngine second;
  ASSERT_EQ(first.load(blob(), 10), Error::Ok);
  ASSERT_EQ(second.load(blob(), 11), Error::Ok);
  ASSERT_TRUE(first.handle()->shared_scratch);
  ASSERT_TRUE(second.handle()->shared_scratch);

  // Held work at the head of the first stream, so the first enqueue and the
  // completion event recorded after it stay pending for as long as the test
  // wants them to.
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(first_stream, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, first_stream);

  // Held for the checks below, which take the watchdog flag first: a call that
  // blocks on the held stream comes back with an error once the watchdog opens
  // the gate, and that error on its own does not say so.
  const Error first_error = first.run(first_stream);
  const Error second_error = second.run(second_stream);

  bool second_finished_early = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    if (cudaStreamQuery(second_stream) == cudaSuccess) {
      second_finished_early = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ASSERT_FALSE(gate.forced_open.load())
      << "the watchdog had to open the gate: a call blocked on the held stream rather than returning, "
         "so nothing below was measured under the conditions it describes";
  ASSERT_EQ(first_error, Error::Ok);
  ASSERT_EQ(second_error, Error::Ok);

  gate_release.release();
  ASSERT_EQ(cudaStreamSynchronize(first_stream), cudaSuccess);
  // Rules out the second engine's work having failed rather than been held,
  // which would leave the check below false for the wrong reason.
  ASSERT_EQ(cudaStreamSynchronize(second_stream), cudaSuccess);

  EXPECT_FALSE(second_finished_early)
      << "the second engine ran to completion while the first one's enqueue was still holding the shared buffer";

  const std::vector<float> first_output = first.read_output();
  const std::vector<float> second_output = second.read_output();
  ASSERT_EQ(first_output.size(), kElems);
  ASSERT_EQ(second_output.size(), kElems);
  EXPECT_NE(std::memcmp(first_output.data(), second_output.data(), kBytes), 0)
      << "the two engines were given different inputs but produced the same output";

  ASSERT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);
}

// ---------------------------------------------------------------------------
// The pooled path under two concurrent callers
// ---------------------------------------------------------------------------

// The window a dropped lock would leave open is a few microseconds wide, so one
// pair of runs would find it only by luck. At this count a build that leaves the
// window open loses most of the runs, and the test costs about two seconds.
constexpr int kConcurrentRunsPerThread = 60;

// Two pooled engines on one device, submitted from two threads on two streams,
// with nothing but the backend ordering them. Both are backed by the same buffer,
// so a claim that ends before the enqueue is recorded hands a second caller the
// same scratch with nothing ordering the two -- silently wrong output, no CUDA
// error, no TensorRT error. Each thread compares byte-for-byte against what its own
// engine produces with private scratch.
TEST_F(SharedScratchBackendTest, TwoThreadsRunningPooledEnginesOnOneDeviceKeepTheirOwnOutputs) {
  cudaStream_t reference_stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&reference_stream), cudaSuccess);

  std::vector<float> first_expected;
  std::vector<float> second_expected;
  {
    LoadedEngine first_priv;
    LoadedEngine second_priv;
    ASSERT_EQ(first_priv.load(blob(), 17), Error::Ok);
    ASSERT_EQ(second_priv.load(blob(), 18), Error::Ok);
    ASSERT_FALSE(first_priv.handle()->shared_scratch);
    ASSERT_FALSE(second_priv.handle()->shared_scratch);
    ASSERT_EQ(first_priv.run(reference_stream), Error::Ok);
    ASSERT_EQ(second_priv.run(reference_stream), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(reference_stream), cudaSuccess);
    first_expected = first_priv.read_output();
    second_expected = second_priv.read_output();
  }
  ASSERT_EQ(cudaStreamDestroy(reference_stream), cudaSuccess);

  ASSERT_EQ(first_expected.size(), kElems);
  ASSERT_EQ(second_expected.size(), kElems);
  // Two engines producing the same bytes would let each thread pass on the other
  // one's output, which is the outcome this test exists to catch.
  ASSERT_NE(std::memcmp(first_expected.data(), second_expected.data(), kBytes), 0)
      << "the two engines were given different inputs but produced the same output";
  ASSERT_NE(first_expected[0], kSentinel) << "the reference output is the sentinel, so a skipped enqueue would pass";
  ASSERT_NE(second_expected[0], kSentinel) << "the reference output is the sentinel, so a skipped enqueue would pass";

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine first;
  LoadedEngine second;
  ASSERT_EQ(first.load(blob(), 17), Error::Ok);
  ASSERT_EQ(second.load(blob(), 18), Error::Ok);
  ASSERT_TRUE(first.handle()->shared_scratch);
  ASSERT_TRUE(second.handle()->shared_scratch);

  cudaStream_t first_stream = nullptr;
  cudaStream_t second_stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&first_stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&second_stream, cudaStreamNonBlocking), cudaSuccess);

  // The host copies that bracket each run synchronize the whole device, so two
  // threads left to themselves take turns rather than overlap. Against a build
  // that leaves the window open, taking turns caught it in 2 of the 120 runs
  // below; releasing both threads together caught nearly all of them. Neither
  // thread can strand the other here -- both run the same fixed
  // number of iterations and neither leaves the loop early.
  std::atomic<int> arrived{0};
  auto submit_together = [&arrived](int iteration) {
    arrived.fetch_add(1);
    while (arrived.load() < 2 * (iteration + 1)) {
      std::this_thread::yield();
    }
  };

  std::atomic<int> wrong_outputs{0};
  std::atomic<int> failures{0};
  auto run_repeatedly = [&](LoadedEngine& engine, const std::vector<float>& expected, cudaStream_t stream) {
    for (int i = 0; i < kConcurrentRunsPerThread; ++i) {
      // Rewritten every iteration, so a run whose enqueue never reached the engine
      // leaves the sentinel behind rather than the previous iteration's output.
      if (!engine.fill_output(kSentinel)) {
        failures.fetch_add(1);
      }
      submit_together(i);
      if (engine.run(stream) != Error::Ok || cudaStreamSynchronize(stream) != cudaSuccess) {
        failures.fetch_add(1);
        continue;
      }
      const std::vector<float> actual = engine.read_output();
      if (actual.size() != expected.size() || std::memcmp(actual.data(), expected.data(), kBytes) != 0) {
        wrong_outputs.fetch_add(1);
      }
    }
  };

  std::thread first_thread([&] { run_repeatedly(first, first_expected, first_stream); });
  std::thread second_thread([&] { run_repeatedly(second, second_expected, second_stream); });
  first_thread.join();
  second_thread.join();

  ASSERT_EQ(cudaStreamDestroy(first_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(second_stream), cudaSuccess);

  EXPECT_EQ(failures.load(), 0) << "a run failed outright, so fewer than " << (2 * kConcurrentRunsPerThread)
                                << " runs reached the comparison below";
  EXPECT_EQ(wrong_outputs.load(), 0) << wrong_outputs.load() << " of " << (2 * kConcurrentRunsPerThread)
                                     << " concurrent pooled runs did not produce what the same engine produces with "
                                        "its own scratch";
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
