/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Handles loaded from the same engine bytes share one deserialized engine, each with its own
// execution context. Each case that expects loads to share checks engine identity, so it fails on a
// backend that deserializes per handle. The engines are built here, which needs a CUDA device: without one
// every SharedEnginesTest case skips, and TORCHTRT_EXECUTORCH_REQUIRE_CUDA=1 turns that skip into a
// failure. If TensorRT cannot build the engines, every fixture case fails. The other suites need no device.

#include "torch_tensorrt/executorch/SharedEngineKey.h"
#include "torch_tensorrt/executorch/SharedEngineTestHooks.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <cuda_runtime.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/core/named_data_map.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

namespace {
thread_local bool fail_memory_query = false;
}

extern "C" cudaError_t __real_cudaMemGetInfo(std::size_t*, std::size_t*);
extern "C" cudaError_t __wrap_cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
  return fail_memory_query ? cudaErrorUnknown : __real_cudaMemGetInfo(free_bytes, total_bytes);
}

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using namespace ::executorch::runtime;
// executorch::aten also has an ArrayRef, so a second using directive would make it ambiguous.
using ::executorch::aten::ScalarType;
using ::executorch::aten::SizesType;

// Spelled out rather than taken from TensorRTBackend.h, so the test pins the key's value.
constexpr char kSharingKey[] = "use_shared_engines";
constexpr char kRequireCudaEnvVar[] = "TORCHTRT_EXECUTORCH_REQUIRE_CUDA";

// A square matmul against a constant matrix: the constant is the weights the engines share, and
// it is large enough for weight streaming to have something to stream.
constexpr int kDim = 512;
constexpr std::size_t kMatElems = static_cast<std::size_t>(kDim) * kDim;
constexpr int kMaxBatch = 64;

// Bounds every wait for another thread, so a partner that never arrives fails the case by name
// instead of hanging the binary until the target's timeout.
constexpr std::chrono::seconds kRendezvousDeadline{30};

class BuilderLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
  }
};

template <typename T>
void write_field(std::vector<std::uint8_t>& blob, std::size_t offset, T value) {
  std::memcpy(blob.data() + offset, &value, sizeof(value));
}

// `extra_binding` is appended to the binding list, so a case can name a binding the engine lacks.
std::string blob_metadata(int device_id, const char* extra_binding = "") {
  return std::string(R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false})") +
      extra_binding + R"(],"hardware_compatible":false,"device_id":)" + std::to_string(device_id) + "}";
}

std::vector<std::uint8_t> wrap_engine_plan(
    const std::uint8_t* plan,
    std::size_t plan_size,
    const std::string& metadata) {
  constexpr std::uint32_t kHeaderSize = 32;
  const auto metadata_size = static_cast<std::uint32_t>(metadata.size());
  const std::uint32_t engine_offset = (kHeaderSize + metadata_size + 15u) / 16u * 16u;
  std::vector<std::uint8_t> blob(static_cast<std::size_t>(engine_offset) + plan_size, 0);
  std::memcpy(blob.data(), "TR01", 4);
  write_field(blob, 4, kHeaderSize);
  write_field(blob, 8, metadata_size);
  write_field(blob, 12, engine_offset);
  write_field(blob, 16, static_cast<std::uint64_t>(plan_size));
  std::memcpy(blob.data() + kHeaderSize, metadata.data(), metadata.size());
  if (plan_size != 0) {
    std::memcpy(blob.data() + engine_offset, plan, plan_size);
  }
  return blob;
}

std::size_t plan_offset(const std::vector<std::uint8_t>& blob) {
  std::uint32_t offset = 0;
  std::memcpy(&offset, blob.data() + 12, sizeof(offset));
  return offset;
}

std::vector<std::uint8_t> rewrap(const std::vector<std::uint8_t>& blob, const std::string& metadata) {
  const std::size_t offset = plan_offset(blob);
  return wrap_engine_plan(blob.data() + offset, blob.size() - offset, metadata);
}

struct BuildOptions {
  float weight_scale = 1.0f;
  bool dynamic_batch = false;
  bool weight_streaming = false;
  bool strip_weights = false;
};

std::vector<float> matmul_weights(float weight_scale) {
  std::vector<float> w(kMatElems);
  for (int i = 0; i < kDim; ++i) {
    for (int j = 0; j < kDim; ++j) {
      w[static_cast<std::size_t>(i) * kDim + j] = weight_scale * static_cast<float>((i * 7 + j * 3) % 17 - 8) / 64.0f;
    }
  }
  return w;
}

// output[b] = relu(input[b] x W), W[i][j] = weight_scale * ((i * 7 + j * 3) % 17 - 8) / 64.
// Two engines that differ only in weight_scale differ only in their weight bytes, so they are the
// same size, which is the pair a size-only key would wrongly share. One builder and one timing
// cache for every build, so both pick the same tactics and the sizes really do match.
std::vector<std::uint8_t> build_matmul_blob(const BuildOptions& options) {
  static BuilderLogger logger;
  static TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  static TRTUniquePtr<nvinfer1::ITimingCache> timing_cache;
  if (builder == nullptr) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  if (network == nullptr) {
    return {};
  }
  const std::int64_t batch = options.dynamic_batch ? -1 : 1;
  nvinfer1::ITensor* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{batch, kDim});
  // Read by buildSerializedNetwork, so it lives until this returns.
  const std::vector<float> w = matmul_weights(options.weight_scale);
  nvinfer1::IConstantLayer* weights = network->addConstant(
      nvinfer1::Dims2{kDim, kDim},
      nvinfer1::Weights{nvinfer1::DataType::kFLOAT, w.data(), static_cast<std::int64_t>(kMatElems)});
  if (input == nullptr || weights == nullptr) {
    return {};
  }
  nvinfer1::IMatrixMultiplyLayer* mm = network->addMatrixMultiply(
      *input, nvinfer1::MatrixOperation::kNONE, *weights->getOutput(0), nvinfer1::MatrixOperation::kNONE);
  nvinfer1::IActivationLayer* relu = network->addActivation(*mm->getOutput(0), nvinfer1::ActivationType::kRELU);
  relu->getOutput(0)->setName("output_0");
  network->markOutput(*relu->getOutput(0));

  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (config == nullptr) {
    return {};
  }
  if (timing_cache == nullptr) {
    timing_cache.reset(config->createTimingCache(nullptr, 0));
  }
  if (timing_cache == nullptr || !config->setTimingCache(*timing_cache, false)) {
    return {};
  }
  if (options.weight_streaming) {
    config->setFlag(nvinfer1::BuilderFlag::kWEIGHT_STREAMING);
  }
  if (options.strip_weights) {
    config->setFlag(nvinfer1::BuilderFlag::kREFIT);
    config->setFlag(nvinfer1::BuilderFlag::kSTRIP_PLAN);
  }
  if (options.dynamic_batch) {
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, kDim});
    profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{8, kDim});
    profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{kMaxBatch, kDim});
    config->addOptimizationProfile(profile);
  }
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (plan == nullptr) {
    return {};
  }
  return wrap_engine_plan(static_cast<const std::uint8_t*>(plan->data()), plan->size(), blob_metadata(0));
}

// load_engine logs this once for each streaming engine it loads, so counting it counts the
// deserializations of a streaming engine.
constexpr std::string_view kBudgetLine = "weight streaming budget=";
std::atomic<int> budget_lines{0};

// While armed, a thread that logs a TensorRT deserialization error waits for this gate, the way a
// log line waits for the interpreter lock under the Python bindings.
std::mutex stall_gate;
std::atomic<bool> stall_armed{false};
std::atomic<bool> logger_stalled{false};
thread_local std::string last_trt_error;
thread_local std::string load_error;
thread_local std::string reported_trt_error;
thread_local const char* next_trt_error = nullptr;

void tap_log(
    et_timestamp_t,
    et_pal_log_level_t level,
    const char*,
    const char*,
    std::size_t,
    const char* message,
    std::size_t length) {
  std::fprintf(stderr, "%c %.*s\n", static_cast<char>(level), static_cast<int>(length), message);
  const std::string_view line(message, length);
  if (line.rfind("TensorRT: ", 0) == 0) {
    last_trt_error = line.substr(std::strlen("TensorRT: "));
    if (next_trt_error != nullptr) {
      const char* replacement = next_trt_error;
      next_trt_error = nullptr;
      TRTLogger logger;
      logger.log(nvinfer1::ILogger::Severity::kERROR, replacement);
    }
  }
  if (line.rfind("TensorRTBackend::init: failed to ", 0) == 0) {
    load_error = line;
    reported_trt_error.clear();
  }
  constexpr std::string_view reason_prefix = "TensorRTBackend::init: TensorRT error: ";
  if (line.rfind(reason_prefix, 0) == 0) {
    reported_trt_error += line.substr(reason_prefix.size());
  }
  if (line.find(kBudgetLine) != std::string_view::npos) {
    budget_lines.fetch_add(1);
  }
  if (stall_armed.load() && line.rfind("TensorRT: ", 0) == 0 &&
      line.find("deserializeCudaEngine") != std::string_view::npos) {
    logger_stalled.store(true);
    const std::lock_guard<std::mutex> wait(stall_gate);
  }
}

bool wait_for_flag(const std::atomic<bool>& flag) {
  const auto deadline = std::chrono::steady_clock::now() + kRendezvousDeadline;
  while (!flag.load()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

float input_value(std::size_t index, std::uint32_t seed) {
  std::uint32_t h = static_cast<std::uint32_t>(index) * 2654435761u + seed * 40503u;
  h ^= h >> 15;
  return static_cast<float>(h % 1000u) / 500.0f - 1.0f;
}

template <typename T>
BackendOption sharing_option(T value) {
  BackendOption option;
  std::strncpy(option.key, kSharingKey, sizeof(option.key) - 1);
  option.value = value;
  return option;
}

Error apply_options(Span<BackendOption> options) {
  BackendOptionContext context;
  TensorRTBackend backend;
  return backend.set_option(context, options);
}

bool wait_for_all(std::atomic<int>& arrived, int count) {
  arrived.fetch_add(1);
  const auto deadline = std::chrono::steady_clock::now() + kRendezvousDeadline;
  while (arrived.load() < count) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

class Handle {
 public:
  Handle() = default;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  ~Handle() {
    if (handle_ != nullptr) {
      backend_.destroy(handle_);
    }
    cudaFree(device_in_);
    cudaFree(device_out_);
  }

  // `budget` is a weight_streaming_budget compile spec, as export bakes it, or null for none.
  // The blob is copied first and the copy overwritten once init returns, so a handle that kept
  // pointing at the bytes it was loaded from would fail.
  Error load(
      const std::vector<std::uint8_t>& blob,
      const char* budget = nullptr,
      Span<const BackendOption> options = {},
      const NamedDataMap* data_map = nullptr) {
    std::vector<std::uint8_t> bytes = blob;
    FreeableBuffer processed(bytes.data(), bytes.size(), nullptr);
    BackendInitContext init_context(&arena_, nullptr, nullptr, data_map, options);
    CompileSpec spec{"weight_streaming_budget", {nullptr, 0}};
    if (budget != nullptr) {
      spec.value.buffer = const_cast<char*>(budget);
      spec.value.nbytes = std::strlen(budget);
    }
    const auto result = backend_.init(
        init_context, &processed, budget != nullptr ? ArrayRef<CompileSpec>(&spec, 1) : ArrayRef<CompileSpec>{});
    std::memset(bytes.data(), 0xA5, bytes.size());
    if (!result.ok()) {
      return result.error();
    }
    handle_ = result.get();
    return Error::Ok;
  }

  std::vector<float> run(int batch, std::uint32_t seed, cudaStream_t stream = nullptr) {
    const std::size_t elems = static_cast<std::size_t>(batch) * kDim;
    if (elems > capacity_) {
      cudaFree(device_in_);
      cudaFree(device_out_);
      device_in_ = device_out_ = nullptr;
      capacity_ = 0;
      if (cudaMalloc(&device_in_, elems * sizeof(float)) != cudaSuccess ||
          cudaMalloc(&device_out_, elems * sizeof(float)) != cudaSuccess) {
        return {};
      }
      capacity_ = elems;
    }
    std::vector<float> host(elems);
    for (std::size_t i = 0; i < elems; ++i) {
      host[i] = input_value(i, seed);
    }
    if (cudaMemcpy(device_in_, host.data(), elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      return {};
    }
    // 0xFF bytes are a NaN, which equals nothing, so a run that never wrote its output cannot pass
    // on what an earlier run left there. Synchronous, so the fill lands before the run starts.
    if (cudaMemset(device_out_, 0xFF, elems * sizeof(float)) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
      return {};
    }
    SizesType in_sizes[2] = {static_cast<SizesType>(batch), kDim};
    SizesType out_sizes[2] = {static_cast<SizesType>(batch), kDim};
    ::executorch::aten::TensorImpl in_impl(ScalarType::Float, 2, in_sizes, device_in_);
    ::executorch::aten::TensorImpl out_impl(ScalarType::Float, 2, out_sizes, device_out_);
    ::executorch::aten::Tensor in_tensor(&in_impl);
    ::executorch::aten::Tensor out_tensor(&out_impl);
    EValue in_value(in_tensor);
    EValue out_value(out_tensor);
    EValue* args[2] = {&in_value, &out_value};
    BackendExecutionContext exec_context;
    Error err;
    if (stream != nullptr) {
      ::executorch::extension::cuda::CallerStreamGuard guard(stream);
      err = backend_.execute(exec_context, handle_, Span<EValue*>(args, 2));
    } else {
      err = backend_.execute(exec_context, handle_, Span<EValue*>(args, 2));
    }
    if (err != Error::Ok || (stream != nullptr && cudaStreamSynchronize(stream) != cudaSuccess)) {
      return {};
    }
    if (cudaMemcpy(host.data(), device_out_, elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
      return {};
    }
    return host;
  }

  const EngineHandle& handle() const {
    return *static_cast<const EngineHandle*>(handle_);
  }

  nvinfer1::ICudaEngine* engine() const {
    return handle().engine.get();
  }

 private:
  TensorRTBackend backend_;
  std::uint8_t arena_storage_[4096];
  MemoryAllocator arena_{sizeof(arena_storage_), arena_storage_};
  DelegateHandle* handle_ = nullptr;
  void* device_in_ = nullptr;
  void* device_out_ = nullptr;
  std::size_t capacity_ = 0;
};

void expect_two_loads_to_share(const std::vector<std::uint8_t>& blob) {
  Handle a;
  Handle b;
  ASSERT_EQ(a.load(blob), Error::Ok);
  ASSERT_EQ(b.load(blob), Error::Ok);
  EXPECT_EQ(a.engine(), b.engine());
}

// One named entry, the way an ExecuTorch .ptd file presents it. Each read gets its own copy, which is
// overwritten when freed, so a backend that read the engine after freeing it would fail.
class OneEntryDataMap final : public NamedDataMap {
 public:
  OneEntryDataMap(std::string key, std::vector<std::uint8_t> data) : key_(std::move(key)), data_(std::move(data)) {}

  Result<const TensorLayout> get_tensor_layout(std::string_view) const override {
    return Error::NotImplemented;
  }
  Result<FreeableBuffer> get_data(std::string_view key) const override {
    reads_.fetch_add(1);
    if (key != key_) {
      return Error::NotFound;
    }
    live_buffers_.fetch_add(1);
    auto* copy = new std::uint8_t[data_.size()];
    std::memcpy(copy, data_.data(), data_.size());
    return FreeableBuffer(
        copy,
        data_.size(),
        [](void* context, void* data, std::size_t size) {
          std::memset(data, 0xA5, size);
          delete[] static_cast<std::uint8_t*>(data);
          static_cast<std::atomic<int>*>(context)->fetch_sub(1);
        },
        &live_buffers_);
  }
  Error load_data_into(std::string_view, void*, std::size_t) const override {
    return Error::NotImplemented;
  }
  Result<std::uint32_t> get_num_keys() const override {
    return 1;
  }
  Result<const char*> get_key(std::uint32_t index) const override {
    if (index != 0) {
      return Error::InvalidArgument;
    }
    return key_.c_str();
  }

  int live_buffers() const {
    return live_buffers_.load();
  }
  int reads() const {
    return reads_.load();
  }

 private:
  std::string key_;
  std::vector<std::uint8_t> data_;
  mutable std::atomic<int> live_buffers_{0};
  mutable std::atomic<int> reads_{0};
};

constexpr char kEngineKey[] = "engine_in_a_data_file";

// The blob export writes for an engine kept in a .ptd file: the metadata names the key, and no
// engine bytes follow it.
std::vector<std::uint8_t> external_engine_blob(const char* key = kEngineKey) {
  std::string metadata = blob_metadata(0);
  metadata.pop_back();
  metadata += std::string(R"(,"engine_key":")") + key + "\"}";
  return wrap_engine_plan(nullptr, 0, metadata);
}

std::vector<std::uint8_t> plan_of(const std::vector<std::uint8_t>& blob) {
  return {blob.begin() + static_cast<std::ptrdiff_t>(plan_offset(blob)), blob.end()};
}

class SharedEnginesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    ::executorch::runtime::runtime_init();
    // Once per process, since a second registration logs an error.
    static const bool tapped = register_pal(PalImpl::create(tap_log, __FILE__));
    (void)tapped;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      return;
    }
    blob_ = build_matmul_blob({});
    other_weights_blob_ = build_matmul_blob({0.5f, false, false});
    dynamic_blob_ = build_matmul_blob({1.0f, true, false});
    streaming_blob_ = build_matmul_blob({1.0f, false, true});
    stripped_blob_ = build_matmul_blob({1.0f, false, false, true});
  }

  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      const char* const required = std::getenv(kRequireCudaEnvVar);
      if (required != nullptr && std::strcmp(required, "1") == 0) {
        FAIL() << "no CUDA device, and " << kRequireCudaEnvVar << "=1 says this run must have one";
      }
      GTEST_SKIP() << "no CUDA device: engine sharing is not covered by this run";
    }
    last_trt_error.clear();
    load_error.clear();
    reported_trt_error.clear();
    ASSERT_FALSE(
        blob_.empty() || other_weights_blob_.empty() || dynamic_blob_.empty() || streaming_blob_.empty() ||
        stripped_blob_.empty())
        << "TensorRT could not build the fixture engines";
  }

  static std::vector<std::uint8_t> blob_;
  static std::vector<std::uint8_t> other_weights_blob_;
  static std::vector<std::uint8_t> dynamic_blob_;
  static std::vector<std::uint8_t> streaming_blob_;
  static std::vector<std::uint8_t> stripped_blob_;
};

std::vector<std::uint8_t> SharedEnginesTest::blob_;
std::vector<std::uint8_t> SharedEnginesTest::other_weights_blob_;
std::vector<std::uint8_t> SharedEnginesTest::dynamic_blob_;
std::vector<std::uint8_t> SharedEnginesTest::streaming_blob_;
std::vector<std::uint8_t> SharedEnginesTest::stripped_blob_;

// Runs without a GPU, so a key that ignored the bytes, the device or the budget fails there too.
// The size changes the hashed bytes as well, so dropping it alone cannot show without a collision.
TEST(SharedEngineKeyTest, TheBytesTheSizeTheDeviceAndTheBudgetEachGiveAnotherKey) {
  const char bytes[] = "engine bytes 1";
  const char same_size[] = "engine bytes 2";
  const std::string copy(bytes, sizeof(bytes));
  const SharedEngineKey key = shared_engine_key(bytes, sizeof(bytes), 0, -1);
  EXPECT_EQ(key, shared_engine_key(copy.data(), copy.size(), 0, -1));
  EXPECT_NE(key, shared_engine_key(same_size, sizeof(same_size), 0, -1));
  EXPECT_NE(key, shared_engine_key(bytes, sizeof(bytes) - 1, 0, -1));
  EXPECT_NE(key, shared_engine_key(bytes, sizeof(bytes), 1, -1));
  EXPECT_NE(key, shared_engine_key(bytes, sizeof(bytes), 0, 0));
}

// The backend's half: it puts the handle's device in the key. Needs a second GPU, because init
// selects the blob's device before anything else.
TEST_F(SharedEnginesTest, TheSamePlanForTwoDevicesLoadsTwoEngines) {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "needs two CUDA devices";
  }
  Handle on_0;
  Handle on_1;
  ASSERT_EQ(on_0.load(blob_), Error::Ok);
  ASSERT_EQ(on_1.load(rewrap(blob_, blob_metadata(1))), Error::Ok);
  EXPECT_NE(on_0.engine(), on_1.engine());
}

TEST_F(SharedEnginesTest, SharingIsOnByDefault) {
  expect_two_loads_to_share(blob_);
}

TEST_F(SharedEnginesTest, TwoLoadsOfTheSameBytesShareOneEngineAndKeepTheirOwnContexts) {
  Handle a;
  Handle b;
  ASSERT_EQ(a.load(blob_), Error::Ok);
  ASSERT_EQ(b.load(blob_), Error::Ok);
  EXPECT_EQ(a.engine(), b.engine());
  EXPECT_NE(a.handle().exec_ctx.get(), b.handle().exec_ctx.get());
  const std::vector<float> out_a = a.run(1, 1);
  const std::vector<float> out_b = b.run(1, 1);
  ASSERT_FALSE(out_a.empty());
  EXPECT_EQ(out_a, out_b);
}

// A hit must not deserialize again: that time and the transient copy of the weights are what
// sharing saves at load.
TEST_F(SharedEnginesTest, ALoadThatFindsALiveEngineDoesNotDeserializeIt) {
  const int before = budget_lines.load();
  Handle first;
  ASSERT_EQ(first.load(streaming_blob_), Error::Ok);
  ASSERT_EQ(budget_lines.load() - before, 1)
      << "a streaming load must log its budget once for this case to count loads";
  Handle second;
  ASSERT_EQ(second.load(streaming_blob_), Error::Ok);
  ASSERT_EQ(second.engine(), first.engine());
  EXPECT_EQ(budget_lines.load() - before, 1) << "the second load deserialized the engine again";
}

// Same size, same I/O, different weights: only the bytes tell them apart.
TEST_F(SharedEnginesTest, EnginesWithDifferentWeightsOfTheSameSizeAreNotShared) {
  ASSERT_EQ(blob_.size(), other_weights_blob_.size()) << "the pair must be the same size to catch a size-only key";
  ASSERT_NE(blob_, other_weights_blob_);
  Handle same_a;
  Handle same_b;
  Handle other;
  ASSERT_EQ(same_a.load(blob_), Error::Ok);
  ASSERT_EQ(same_b.load(blob_), Error::Ok);
  ASSERT_EQ(other.load(other_weights_blob_), Error::Ok);
  ASSERT_EQ(same_a.engine(), same_b.engine());
  EXPECT_NE(same_a.engine(), other.engine());
  const std::vector<float> out = same_a.run(1, 3);
  const std::vector<float> out_other = other.run(1, 3);
  ASSERT_FALSE(out.empty());
  ASSERT_FALSE(out_other.empty());
  EXPECT_NE(out, out_other);
}

// A plan without its weights runs without an error and returns wrong results, so it has to fail the
// load. Sharing on and off are two paths into the deserializer, and both must refuse it. The same
// engine serialized with its weights is the control: it loads and matches the ordinary engine.
TEST_F(SharedEnginesTest, APlanSerializedWithoutItsWeightsIsRefused) {
  const BackendOption private_load = sharing_option(false);
  Handle shared;
  Handle unshared;
  EXPECT_EQ(shared.load(stripped_blob_), Error::InvalidProgram);
  EXPECT_EQ(unshared.load(stripped_blob_, nullptr, {&private_load, 1}), Error::InvalidProgram);

  static BuilderLogger logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  const std::size_t offset = plan_offset(stripped_blob_);
  TRTUniquePtr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(stripped_blob_.data() + offset, stripped_blob_.size() - offset));
  ASSERT_NE(engine, nullptr);
  ASSERT_GE(engine->getEngineStat(nvinfer1::EngineStat::kSTRIPPED_WEIGHTS_SIZE), 0);
  TRTUniquePtr<nvinfer1::IRefitter> refitter(nvinfer1::createInferRefitter(*engine, logger));
  ASSERT_NE(refitter, nullptr);
  const char* name = nullptr;
  ASSERT_EQ(refitter->getAllWeights(1, &name), 1) << "the matmul has one refittable weight";
  const std::vector<float> w = matmul_weights(1.0f);
  ASSERT_TRUE(refitter->setNamedWeights(
      name, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, w.data(), static_cast<std::int64_t>(kMatElems)}));
  ASSERT_TRUE(refitter->refitCudaEngine());
  TRTUniquePtr<nvinfer1::ISerializationConfig> config(engine->createSerializationConfig());
  config->clearFlag(nvinfer1::SerializationFlag::kEXCLUDE_WEIGHTS);
  config->setFlag(nvinfer1::SerializationFlag::kINCLUDE_REFIT);
  TRTUniquePtr<nvinfer1::IHostMemory> full(engine->serializeWithConfig(*config));
  ASSERT_NE(full, nullptr);
  const auto with_weights =
      wrap_engine_plan(static_cast<const std::uint8_t*>(full->data()), full->size(), blob_metadata(0));

  // Two builds may pick different kernels, so they agree only within rounding. A plan computing with
  // placeholder weights misses by whole units.
  Handle reference;
  Handle restored;
  ASSERT_EQ(reference.load(blob_), Error::Ok);
  ASSERT_EQ(restored.load(with_weights), Error::Ok);
  const auto expected = reference.run(1, 7);
  const auto actual = restored.run(1, 7);
  ASSERT_FALSE(expected.empty());
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    ASSERT_NEAR(actual[i], expected[i], 1e-2f * (1.0f + std::abs(expected[i]))) << "element " << i;
  }
}

TEST_F(SharedEnginesTest, TheEngineIsFreedWithItsLastHandle) {
  std::weak_ptr<nvinfer1::ICudaEngine> watched;
  {
    Handle a;
    ASSERT_EQ(a.load(blob_), Error::Ok);
    watched = a.handle().engine;
    {
      Handle b;
      ASSERT_EQ(b.load(blob_), Error::Ok);
      ASSERT_EQ(b.engine(), a.engine());
    }
    EXPECT_FALSE(watched.expired()) << "the first handle still holds the engine";
    EXPECT_FALSE(a.run(1, 5).empty()) << "the surviving handle still runs after its peer is destroyed";
  }
  EXPECT_TRUE(watched.expired()) << "nothing but the handles may keep the engine alive";
  Handle reloaded;
  ASSERT_EQ(reloaded.load(blob_), Error::Ok);
  EXPECT_FALSE(reloaded.run(1, 5).empty());
}

// The same engine bytes, read from a data file instead of the blob, give the same engine: it runs to
// the same result bit for bit, and a load of the embedded blob shares it.
TEST_F(SharedEnginesTest, AnEngineInADataFileRunsLikeTheSameEngineInTheBlob) {
  const OneEntryDataMap data_map(kEngineKey, plan_of(blob_));
  Handle external;
  ASSERT_EQ(external.load(external_engine_blob(), nullptr, {}, &data_map), Error::Ok);
  EXPECT_EQ(data_map.live_buffers(), 0) << "init must free the engine bytes once the engine is built";
  Handle embedded;
  ASSERT_EQ(embedded.load(blob_), Error::Ok);
  EXPECT_EQ(external.engine(), embedded.engine());
  const auto expected = embedded.run(1, 11);
  ASSERT_FALSE(expected.empty());
  EXPECT_EQ(external.run(1, 11), expected);
  const BackendOption private_load = sharing_option(false);
  Handle private_external;
  ASSERT_EQ(private_external.load(external_engine_blob(), nullptr, {&private_load, 1}, &data_map), Error::Ok);
  EXPECT_EQ(data_map.live_buffers(), 0);
  EXPECT_EQ(private_external.run(1, 11), expected);
}

TEST_F(SharedEnginesTest, AnEngineInADataFileFailsClearlyWithoutTheFile) {
  Handle no_file;
  EXPECT_EQ(no_file.load(external_engine_blob()), Error::InvalidExternalData);
  const OneEntryDataMap other_file("some_other_key", plan_of(blob_));
  Handle wrong_file;
  EXPECT_EQ(wrong_file.load(external_engine_blob(), nullptr, {}, &other_file), Error::InvalidExternalData);
}

// The guard reads the engine, so it applies wherever the bytes came from.
TEST_F(SharedEnginesTest, AStrippedEngineInADataFileIsRefused) {
  const OneEntryDataMap data_map(kEngineKey, plan_of(stripped_blob_));
  Handle handle;
  EXPECT_EQ(handle.load(external_engine_blob(), nullptr, {}, &data_map), Error::InvalidProgram);
  EXPECT_EQ(data_map.reads(), 1) << "refused before the engine was read, so the guard never ran";
  EXPECT_EQ(data_map.live_buffers(), 0);
}

TEST_F(SharedEnginesTest, ABlobNamingAnExternalEngineMustCarryNoEngineBytes) {
  std::string metadata = blob_metadata(0);
  metadata.pop_back();
  metadata += std::string(R"(,"engine_key":")") + kEngineKey + "\"}";
  const auto plan = plan_of(blob_);
  const auto both = wrap_engine_plan(plan.data(), plan.size(), metadata);
  const OneEntryDataMap data_map(kEngineKey, plan);
  Handle handle;
  EXPECT_EQ(handle.load(both, nullptr, {}, &data_map), Error::InvalidProgram);
}

// The budget is fixed before the first context and TensorRT refuses to move it after, so loads
// asking for different budgets need their own engines, and a load asking for the same budget must
// not try to set it again on a shared engine.
TEST_F(SharedEnginesTest, DifferentWeightStreamingBudgetsDoNotShareAndEqualOnesDo) {
  Handle probe;
  ASSERT_EQ(probe.load(streaming_blob_), Error::Ok);
  const std::int64_t streamable = probe.engine()->getStreamableWeightsSize();
  ASSERT_GT(streamable, 0) << "the fixture engine must stream weights for this case to mean anything";
  const std::string half = std::to_string(streamable / 2);
  const std::string quarter = std::to_string(streamable / 4);

  Handle half_a;
  Handle half_b;
  Handle quarter_a;
  ASSERT_EQ(half_a.load(streaming_blob_, half.c_str()), Error::Ok);
  ASSERT_EQ(half_b.load(streaming_blob_, half.c_str()), Error::Ok);
  ASSERT_EQ(quarter_a.load(streaming_blob_, quarter.c_str()), Error::Ok);
  EXPECT_EQ(half_a.engine(), half_b.engine());
  EXPECT_NE(half_a.engine(), quarter_a.engine());
  EXPECT_NE(half_a.engine(), probe.engine()) << "an explicit budget does not share with the automatic one";
  EXPECT_EQ(half_a.engine()->getWeightStreamingBudgetV2(), streamable / 2);
  EXPECT_EQ(quarter_a.engine()->getWeightStreamingBudgetV2(), streamable / 4);
  // A load with no budget also carries 0 bytes, so only the -1 in the key keeps it apart from this.
  Handle zero;
  ASSERT_EQ(zero.load(streaming_blob_, "0"), Error::Ok);
  EXPECT_NE(zero.engine(), probe.engine()) << "an explicit budget of 0 shared the automatic one";
  EXPECT_EQ(zero.engine()->getWeightStreamingBudgetV2(), 0);

  const std::vector<float> reference = probe.run(1, 9);
  ASSERT_FALSE(reference.empty());
  EXPECT_EQ(half_b.run(1, 9), reference);
  EXPECT_EQ(quarter_a.run(1, 9), reference);
}

// The default for a streaming engine: no budget asked, so the first load picks TensorRT's automatic
// one, and a second load shares that engine without setting a budget on it again.
TEST_F(SharedEnginesTest, TwoLoadsWithTheAutomaticBudgetShareOneStreamingEngineAndKeepTheFirstBudget) {
  Handle first;
  ASSERT_EQ(first.load(streaming_blob_), Error::Ok);
  ASSERT_GT(first.engine()->getStreamableWeightsSize(), 0)
      << "the fixture engine must stream weights for this case to mean anything";
  const std::int64_t budget = first.engine()->getWeightStreamingBudgetV2();
  Handle second;
  ASSERT_EQ(second.load(streaming_blob_), Error::Ok);
  EXPECT_EQ(second.engine(), first.engine());
  EXPECT_EQ(second.engine()->getWeightStreamingBudgetV2(), budget);
  const std::vector<float> out = first.run(1, 8);
  ASSERT_FALSE(out.empty());
  EXPECT_EQ(second.run(1, 8), out);
}

// The plan is kept and the metadata names one binding more than the engine has, so the bad load
// shares the good load's engine and fails after that. It must not keep its reference.
TEST_F(SharedEnginesTest, AFailedLoadLeavesNothingBehindForTheLoadsAfterIt) {
  Handle good;
  ASSERT_EQ(good.load(blob_), Error::Ok);
  Handle bad;
  EXPECT_EQ(
      bad.load(rewrap(blob_, blob_metadata(0, R"(,{"name":"output_1","is_input":false})"))), Error::InvalidProgram);
  EXPECT_EQ(good.handle().engine.use_count(), 1) << "the failed load still holds the engine";
  Handle shared;
  ASSERT_EQ(shared.load(blob_), Error::Ok);
  EXPECT_EQ(shared.engine(), good.engine());
  const std::vector<float> out = good.run(1, 6);
  ASSERT_FALSE(out.empty());
  EXPECT_EQ(shared.run(1, 6), out);
}

// The map lock is never held while TensorRT runs: a load stalled in a TensorRT log line, as one
// waiting for Python's interpreter lock would be, must not stop a load on another thread.
TEST_F(SharedEnginesTest, ALoadFinishesWhileAnotherIsStalledInATensorRTLogLine) {
  std::vector<std::uint8_t> damaged = blob_;
  const std::size_t offset = plan_offset(damaged);
  std::memset(damaged.data() + offset, 0, damaged.size() - offset);
  // Kept live so the other load shares it rather than deserializing: this case tests the map lock,
  // not whether TensorRT can deserialize while another thread is stalled in its logger.
  Handle holder;
  ASSERT_EQ(holder.load(blob_), Error::Ok);
  Handle bad;
  Handle good;
  Error bad_result = Error::Ok;
  std::promise<Error> loaded;
  std::future<Error> good_result = loaded.get_future();
  std::unique_lock<std::mutex> gate(stall_gate);
  logger_stalled.store(false);
  stall_armed.store(true);
  std::thread stalled([&] {
    cudaSetDevice(0);
    bad_result = bad.load(damaged);
  });
  const bool reached_log = wait_for_flag(logger_stalled);
  std::thread loader([&] {
    cudaSetDevice(0);
    loaded.set_value(good.load(blob_));
  });
  // Bounded, so a load that does wait on the stalled one fails the case instead of hanging it.
  const bool finished = reached_log && good_result.wait_for(kRendezvousDeadline) == std::future_status::ready;
  stall_armed.store(false);
  gate.unlock();
  stalled.join();
  loader.join();
  ASSERT_TRUE(reached_log) << "TensorRT did not log while deserializing the damaged plan";
  EXPECT_TRUE(finished) << "a load waited for a load stalled in a TensorRT log line";
  EXPECT_EQ(good_result.get(), Error::Ok);
  EXPECT_EQ(good.engine(), holder.engine());
  EXPECT_EQ(bad_result, Error::InvalidProgram);
}

TEST_F(SharedEnginesTest, ConcurrentLoadsOfTheSameBytesEndWithOneEngine) {
  constexpr int kThreads = 4;
  Handle handles[kThreads];
  std::atomic<int> arrived{0};
  std::atomic<bool> partner_never_arrived{false};
  std::vector<Error> results(kThreads, Error::Internal);
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      cudaSetDevice(0);
      if (!wait_for_all(arrived, kThreads)) {
        partner_never_arrived.store(true);
        return;
      }
      results[i] = handles[i].load(blob_);
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  ASSERT_FALSE(partner_never_arrived.load()) << "a thread waited out the rendezvous deadline";
  for (int i = 0; i < kThreads; ++i) {
    ASSERT_EQ(results[i], Error::Ok) << "thread " << i;
  }
  // Racing loads may each deserialize, but every handle ends on the engine that was published.
  for (int i = 1; i < kThreads; ++i) {
    EXPECT_EQ(handles[i].engine(), handles[0].engine()) << "thread " << i;
  }
  EXPECT_EQ(handles[0].handle().engine.use_count(), kThreads);
  Handle later;
  ASSERT_EQ(later.load(blob_), Error::Ok);
  EXPECT_EQ(later.engine(), handles[0].engine());
}

// Two contexts of one engine on two threads and two streams, at once, each with its own input.
void run_two_handles_at_once(Handle& a, Handle& b, int batch_a, int batch_b) {
  const std::vector<float> golden_a = a.run(batch_a, 11);
  const std::vector<float> golden_b = b.run(batch_b, 22);
  ASSERT_FALSE(golden_a.empty());
  ASSERT_FALSE(golden_b.empty());
  // Made here, so a stream that cannot be created fails the case before either thread waits.
  cudaStream_t stream_a = nullptr;
  cudaStream_t stream_b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream_a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream_b, cudaStreamNonBlocking), cudaSuccess);
  std::atomic<int> mismatches{0};
  std::atomic<int> arrived{0};
  std::atomic<bool> partner_never_arrived{false};
  auto worker = [&](Handle& h, int batch, std::uint32_t seed, const std::vector<float>& golden, cudaStream_t stream) {
    cudaSetDevice(0);
    if (!wait_for_all(arrived, 2)) {
      partner_never_arrived.store(true);
      return;
    }
    for (int i = 0; i < 50; ++i) {
      if (h.run(batch, seed, stream) != golden) {
        mismatches.fetch_add(1);
      }
    }
  };
  std::thread ta(worker, std::ref(a), batch_a, 11u, std::cref(golden_a), stream_a);
  std::thread tb(worker, std::ref(b), batch_b, 22u, std::cref(golden_b), stream_b);
  ta.join();
  tb.join();
  EXPECT_EQ(cudaStreamDestroy(stream_a), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream_b), cudaSuccess);
  EXPECT_FALSE(partner_never_arrived.load()) << "a thread waited out the rendezvous deadline";
  EXPECT_EQ(mismatches.load(), 0);
}

TEST_F(SharedEnginesTest, TwoContextsOfOneEngineRunAtOnceOnTwoStreams) {
  Handle a;
  Handle b;
  ASSERT_EQ(a.load(blob_), Error::Ok);
  ASSERT_EQ(b.load(blob_), Error::Ok);
  ASSERT_EQ(a.engine(), b.engine());
  run_two_handles_at_once(a, b, 1, 1);
}

// Every context starts on optimization profile 0. TensorRT has shared profiles between concurrent
// contexts by default since 10.0, which is what lets a dynamic engine be shared at all.
TEST_F(SharedEnginesTest, TwoContextsOfOneDynamicShapeEngineRunAtOnceWithDifferentShapes) {
  Handle a;
  Handle b;
  ASSERT_EQ(a.load(dynamic_blob_), Error::Ok);
  ASSERT_EQ(b.load(dynamic_blob_), Error::Ok);
  ASSERT_EQ(a.engine(), b.engine());
  run_two_handles_at_once(a, b, 3, kMaxBatch);
}

TEST_F(SharedEnginesTest, OneLoadCanStayPrivateWhileOtherLoadsShare) {
  Handle shared;
  ASSERT_EQ(shared.load(blob_), Error::Ok);
  const BackendOption option = sharing_option(false);
  Handle private_a;
  Handle private_b;
  ASSERT_EQ(private_a.load(blob_, nullptr, {&option, 1}), Error::Ok);
  ASSERT_EQ(private_b.load(blob_, nullptr, {&option, 1}), Error::Ok);
  EXPECT_NE(private_a.engine(), shared.engine());
  EXPECT_NE(private_a.engine(), private_b.engine());
  Handle shared_again;
  ASSERT_EQ(shared_again.load(blob_), Error::Ok);
  EXPECT_EQ(shared_again.engine(), shared.engine()) << "a private load took the place of the published engine";
  const auto reference = shared.run(1, 4);
  ASSERT_FALSE(reference.empty());
  EXPECT_EQ(private_a.run(1, 4), reference);
}

TEST_F(SharedEnginesTest, AnEngineLoadedWithSharingOffIsNotSharedLater) {
  const BackendOption option = sharing_option(false);
  Handle private_handle;
  ASSERT_EQ(private_handle.load(blob_, nullptr, {&option, 1}), Error::Ok);
  Handle a;
  Handle b;
  ASSERT_EQ(a.load(blob_), Error::Ok);
  ASSERT_EQ(b.load(blob_), Error::Ok);
  EXPECT_NE(a.engine(), private_handle.engine()) << "an engine loaded with sharing off was published";
  EXPECT_EQ(a.engine(), b.engine());
}

TEST_F(SharedEnginesTest, LoadRejectsANonBooleanSharingOption) {
  const BackendOption option = sharing_option(0);
  Handle invalid;
  EXPECT_EQ(invalid.load(blob_, nullptr, {&option, 1}), Error::InvalidArgument);
  expect_two_loads_to_share(blob_);
}

TEST(SharedEngineOptionsTest, SetOptionRejectsTheLoadOnlySharingKey) {
  ::executorch::runtime::runtime_init();
  for (bool enabled : {false, true}) {
    BackendOption option = sharing_option(enabled);
    EXPECT_EQ(::executorch::runtime::set_option("TensorRTBackend", {&option, 1}), Error::InvalidArgument);
  }
}

TEST_F(SharedEnginesTest, RejectingTheLoadOnlyKeyDoesNotApplyOtherOptions) {
  BackendOption scratch;
  std::strncpy(scratch.key, kSharedActivationScratchKey, sizeof(scratch.key) - 1);
  scratch.value = false;
  ASSERT_EQ(apply_options({&scratch, 1}), Error::Ok);
  struct ResetScratch {
    BackendOption& option;
    ~ResetScratch() {
      option.value = false;
      apply_options({&option, 1});
    }
  } reset{scratch};
  scratch.value = true;
  BackendOption options[] = {scratch, sharing_option(false)};
  EXPECT_EQ(::executorch::runtime::set_option("TensorRTBackend", {options, 2}), Error::InvalidArgument);
  Handle handle;
  ASSERT_EQ(handle.load(blob_), Error::Ok);
  EXPECT_FALSE(handle.handle().shared_scratch);
}

TEST_F(SharedEnginesTest, ConcurrentLoadsKeepTheirOwnSharingOptions) {
  Handle published;
  ASSERT_EQ(published.load(blob_), Error::Ok);
  constexpr int kThreads = 4;
  Handle handles[kThreads];
  Error results[kThreads] = {};
  std::atomic<int> arrived{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([&, i] {
      cudaSetDevice(0);
      if (!wait_for_all(arrived, kThreads)) {
        results[i] = Error::Internal;
        return;
      }
      const BackendOption option = sharing_option(i % 2 != 0);
      results[i] = handles[i].load(blob_, nullptr, {&option, 1});
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  for (int i = 0; i < kThreads; ++i) {
    ASSERT_EQ(results[i], Error::Ok);
    EXPECT_EQ(handles[i].engine() == published.engine(), i % 2 != 0);
  }
  EXPECT_NE(handles[0].engine(), handles[2].engine());
}

TEST_F(SharedEnginesTest, ScratchAndEngineSharingUseSeparateKeys) {
  BackendOption scratch;
  std::strncpy(scratch.key, kSharedActivationScratchKey, sizeof(scratch.key) - 1);
  scratch.value = true;
  ASSERT_EQ(apply_options({&scratch, 1}), Error::Ok);
  struct ResetScratch {
    ~ResetScratch() {
      BackendOption option;
      std::strncpy(option.key, kSharedActivationScratchKey, sizeof(option.key) - 1);
      option.value = false;
      apply_options({&option, 1});
    }
  } reset;
  Handle a;
  Handle b;
  Handle private_handle;
  ASSERT_EQ(a.load(blob_), Error::Ok);
  const BackendOption sharing = sharing_option(false);
  ASSERT_EQ(private_handle.load(blob_, nullptr, {&sharing, 1}), Error::Ok);
  ASSERT_EQ(b.load(blob_), Error::Ok);
  EXPECT_EQ(a.engine(), b.engine());
  EXPECT_NE(a.engine(), private_handle.engine());
  EXPECT_TRUE(a.handle().shared_scratch);
  EXPECT_TRUE(private_handle.handle().shared_scratch);
  scratch.value = false;
  ASSERT_EQ(apply_options({&scratch, 1}), Error::Ok);
  Handle c;
  ASSERT_EQ(c.load(blob_), Error::Ok);
  EXPECT_EQ(c.engine(), a.engine());
  EXPECT_FALSE(c.handle().shared_scratch);
}

TEST_F(SharedEnginesTest, LoadTimeBudgetsSeparateEnginesAndOverrideCompileSpecs) {
  Handle probe;
  ASSERT_EQ(probe.load(streaming_blob_), Error::Ok);
  const std::int64_t streamable = probe.engine()->getStreamableWeightsSize();
  ASSERT_GT(streamable, 0);
  BackendOption half;
  std::strncpy(half.key, "weight_streaming_budget", sizeof(half.key) - 1);
  std::array<char, kMaxOptionValueLength> value{};
  const std::string half_bytes = std::to_string(streamable / 2);
  std::strncpy(value.data(), half_bytes.c_str(), value.size() - 1);
  half.value = value;
  BackendOption quarter = half;
  const std::string quarter_bytes = std::to_string(streamable / 4);
  value.fill(0);
  std::strncpy(value.data(), quarter_bytes.c_str(), value.size() - 1);
  quarter.value = value;
  Handle a;
  Handle b;
  Handle c;
  Handle d;
  Handle e;
  ASSERT_EQ(a.load(streaming_blob_, nullptr, {&half, 1}), Error::Ok);
  ASSERT_EQ(b.load(streaming_blob_, nullptr, {&quarter, 1}), Error::Ok);
  ASSERT_EQ(c.load(streaming_blob_, nullptr, {&half, 1}), Error::Ok);
  ASSERT_EQ(d.load(streaming_blob_, "0", {&half, 1}), Error::Ok);
  ASSERT_EQ(e.load(streaming_blob_, "0", {&quarter, 1}), Error::Ok);
  EXPECT_NE(a.engine(), b.engine());
  EXPECT_NE(a.engine(), probe.engine());
  EXPECT_EQ(a.engine(), c.engine());
  EXPECT_EQ(a.engine(), d.engine());
  EXPECT_EQ(b.engine(), e.engine());
  EXPECT_EQ(a.engine()->getWeightStreamingBudgetV2(), streamable / 2);
  EXPECT_EQ(b.engine()->getWeightStreamingBudgetV2(), streamable / 4);
  const auto reference = probe.run(1, 9);
  ASSERT_FALSE(reference.empty());
  EXPECT_EQ(d.run(1, 9), reference);
  EXPECT_EQ(e.run(1, 9), reference);
}

class FailingAllocator : public nvinfer1::IGpuAllocator {
 public:
  void* allocate(std::uint64_t size, std::uint64_t, nvinfer1::AllocatorFlags) noexcept override {
    if (fail) {
      ++failures;
      return nullptr;
    }
    void* memory = nullptr;
    return cudaMalloc(&memory, size) == cudaSuccess ? memory : nullptr;
  }
  bool deallocate(void* memory) noexcept override {
    return cudaFree(memory) == cudaSuccess;
  }
  bool fail = false;
  std::atomic<int> failures{0};
};

class ScopedGpuAllocator {
 public:
  explicit ScopedGpuAllocator(nvinfer1::IGpuAllocator& allocator) {
    shared_runtime_for_testing()->setGpuAllocator(&allocator);
  }
  ~ScopedGpuAllocator() {
    shared_runtime_for_testing()->setGpuAllocator(nullptr);
  }
};

void expect_load_failure_diagnostics(const char* operation) {
  EXPECT_NE(load_error.find(operation), std::string::npos) << load_error;
  ASSERT_FALSE(last_trt_error.empty());
  // The original TensorRT log line may be truncated by the runtime's log buffer.
  EXPECT_EQ(reported_trt_error.find(last_trt_error), 0) << reported_trt_error;
  const auto value_pos = load_error.find("free=");
  ASSERT_NE(value_pos, std::string::npos) << load_error;
  std::size_t free_bytes = 0;
  std::size_t total = 0;
  ASSERT_EQ(std::sscanf(load_error.c_str() + value_pos, "free=%zu, total=%zu", &free_bytes, &total), 2);
  EXPECT_GT(total, 0);
  EXPECT_LE(free_bytes, total);
  EXPECT_EQ(load_error.find("engine_size="), std::string::npos) << load_error;
}

TEST_F(SharedEnginesTest, EngineAllocationFailureReportsMemoryAndTensorRTReason) {
  ASSERT_NE(shared_runtime_for_testing(), nullptr);
  FailingAllocator allocator;
  ScopedGpuAllocator installed(allocator);
  Handle handle;
  allocator.fail = true;
  last_trt_error.clear();
  load_error.clear();
  EXPECT_EQ(handle.load(blob_), Error::MemoryAllocationFailed);
  ASSERT_GT(allocator.failures.load(), 0);
  expect_load_failure_diagnostics("deserialize TensorRT engine");
}

TEST_F(SharedEnginesTest, ContextAllocationFailureReportsMemoryAndTensorRTReason) {
  ASSERT_NE(shared_runtime_for_testing(), nullptr);
  FailingAllocator allocator;
  ScopedGpuAllocator installed(allocator);
  Handle first;
  ASSERT_EQ(first.load(blob_), Error::Ok);
  Handle second;
  allocator.fail = true;
  last_trt_error.clear();
  load_error.clear();
  EXPECT_EQ(second.load(blob_), Error::MemoryAllocationFailed);
  ASSERT_GT(allocator.failures, 0);
  expect_load_failure_diagnostics("create TensorRT execution context");
}

TEST_F(SharedEnginesTest, TruncatedEngineReportsInvalidProgramAndTensorRTReason) {
  const auto damaged = wrap_engine_plan(blob_.data() + plan_offset(blob_), 32, blob_metadata(0));
  Handle handle;
  last_trt_error.clear();
  load_error.clear();
  EXPECT_EQ(handle.load(damaged), Error::InvalidProgram);
  ASSERT_FALSE(last_trt_error.empty());
  EXPECT_NE(load_error.find("deserialize TensorRT engine"), std::string::npos) << load_error;
  EXPECT_EQ(reported_trt_error.find(last_trt_error), 0) << reported_trt_error;
}

TEST_F(SharedEnginesTest, AnotherThreadsErrorDoesNotChangeTheLoadFailure) {
  const auto damaged = wrap_engine_plan(blob_.data() + plan_offset(blob_), 32, blob_metadata(0));
  Error result = Error::Ok;
  std::string reason;
  std::string reported;
  std::unique_lock<std::mutex> gate(stall_gate);
  logger_stalled.store(false);
  stall_armed.store(true);
  std::thread loader([&] {
    cudaSetDevice(0);
    Handle handle;
    result = handle.load(damaged);
    reason = last_trt_error;
    reported = reported_trt_error;
  });
  const bool reached_log = wait_for_flag(logger_stalled);
  TRTLogger logger;
  logger.log(nvinfer1::ILogger::Severity::kERROR, "out of memory on another thread");
  stall_armed.store(false);
  gate.unlock();
  loader.join();
  ASSERT_TRUE(reached_log);
  EXPECT_EQ(result, Error::InvalidProgram);
  ASSERT_FALSE(reason.empty());
  EXPECT_EQ(reported.find(reason), 0) << reported;
  EXPECT_EQ(reported.find("another thread"), std::string::npos) << reported;
}

TEST_F(SharedEnginesTest, FreshLoadPreservesTheCallersPendingCudaError) {
  Handle handle;
  void* memory = nullptr;
  ASSERT_EQ(cudaMalloc(&memory, std::size_t{1} << 62), cudaErrorMemoryAllocation);
  ASSERT_EQ(cudaPeekAtLastError(), cudaErrorMemoryAllocation);
  EXPECT_EQ(handle.load(blob_), Error::Ok);
  EXPECT_EQ(cudaGetLastError(), cudaErrorMemoryAllocation);
}

TEST_F(SharedEnginesTest, SharedLoadPreservesTheCallersPendingCudaError) {
  Handle first;
  ASSERT_EQ(first.load(blob_), Error::Ok);
  Handle second;
  void* memory = nullptr;
  ASSERT_EQ(cudaMalloc(&memory, std::size_t{1} << 62), cudaErrorMemoryAllocation);
  ASSERT_EQ(cudaPeekAtLastError(), cudaErrorMemoryAllocation);
  EXPECT_EQ(second.load(blob_), Error::Ok);
  EXPECT_EQ(cudaGetLastError(), cudaErrorMemoryAllocation);
}

TEST_F(SharedEnginesTest, FailedLoadPreservesTheCallersPendingCudaError) {
  const auto damaged = wrap_engine_plan(blob_.data() + plan_offset(blob_), 32, blob_metadata(0));
  Handle handle;
  void* memory = nullptr;
  ASSERT_EQ(cudaMalloc(&memory, std::size_t{1} << 62), cudaErrorMemoryAllocation);
  ASSERT_EQ(cudaPeekAtLastError(), cudaErrorMemoryAllocation);
  EXPECT_EQ(handle.load(damaged), Error::InvalidProgram);
  EXPECT_EQ(cudaGetLastError(), cudaErrorMemoryAllocation);
}

TEST_F(SharedEnginesTest, WeightStreamingAllocationFailureReportsMemoryAndTensorRTReason) {
  Handle probe;
  ASSERT_EQ(probe.load(streaming_blob_, "0"), Error::Ok);
  const auto streamable = probe.engine()->getStreamableWeightsSize();
  ASSERT_GT(streamable, 0);
  const std::string budget = std::to_string(streamable);
  FailingAllocator allocator;
  ScopedGpuAllocator installed(allocator);
  Handle handle;
  allocator.fail = true;
  EXPECT_EQ(handle.load(streaming_blob_, budget.c_str()), Error::MemoryAllocationFailed);
  ASSERT_GT(allocator.failures.load(), 0);
  expect_load_failure_diagnostics("set TensorRT weight streaming budget");
}

TEST_F(SharedEnginesTest, AllocationFailureReportsAnUnavailableMemoryQuery) {
  FailingAllocator allocator;
  ScopedGpuAllocator installed(allocator);
  Handle handle;
  allocator.fail = true;
  void* memory = nullptr;
  ASSERT_EQ(cudaMalloc(&memory, std::size_t{1} << 62), cudaErrorMemoryAllocation);
  fail_memory_query = true;
  const auto result = handle.load(blob_);
  fail_memory_query = false;
  EXPECT_EQ(cudaGetLastError(), cudaErrorMemoryAllocation);
  EXPECT_EQ(result, Error::MemoryAllocationFailed);
  ASSERT_GT(allocator.failures.load(), 0);
  EXPECT_NE(load_error.find("device memory unavailable: unknown error"), std::string::npos) << load_error;
  EXPECT_EQ(load_error.find("free="), std::string::npos) << load_error;
  ASSERT_FALSE(last_trt_error.empty());
  EXPECT_EQ(reported_trt_error.find(last_trt_error), 0) << reported_trt_error;
}

TEST_F(SharedEnginesTest, TheLastReasonAndItsErrorCodeStayConsistent) {
  FailingAllocator allocator;
  ScopedGpuAllocator installed(allocator);
  Handle handle;
  allocator.fail = true;
  constexpr char reason[] = "a later general TensorRT failure";
  next_trt_error = reason;
  const auto result = handle.load(blob_);
  next_trt_error = nullptr;
  EXPECT_EQ(result, Error::InvalidProgram);
  ASSERT_GT(allocator.failures.load(), 0);
  EXPECT_EQ(reported_trt_error, reason);
  EXPECT_EQ(load_error.find("out of memory"), std::string::npos) << load_error;
}

TEST_F(SharedEnginesTest, LongTensorRTReasonIsPreservedAcrossLogLines) {
  std::vector<std::uint8_t> damaged = blob_;
  const auto offset = plan_offset(damaged);
  std::memset(damaged.data() + offset, 0, damaged.size() - offset);
  class RecordingLogger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, const char* message) noexcept override {
      if (severity <= Severity::kERROR) {
        reason = message;
      }
    }
    std::string reason;
  } logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  ASSERT_NE(runtime, nullptr);
  TRTUniquePtr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(damaged.data() + offset, damaged.size() - offset));
  ASSERT_EQ(engine, nullptr);
  ASSERT_GT(logger.reason.size(), 200);
  ASSERT_LT(logger.reason.size(), 1024);
  Handle handle;
  EXPECT_EQ(handle.load(damaged), Error::InvalidProgram);
  EXPECT_EQ(reported_trt_error, logger.reason);
}

thread_local bool fail_this_load = false;

class GatingAllocator : public nvinfer1::IGpuAllocator {
 public:
  void* allocate(std::uint64_t size, std::uint64_t, nvinfer1::AllocatorFlags) noexcept override {
    if (fail_this_load && armed.exchange(false)) {
      std::unique_lock<std::mutex> lock(mutex);
      entered.store(true);
      released.wait_for(lock, kRendezvousDeadline, [&] { return proceed; });
      return nullptr;
    }
    void* memory = nullptr;
    return cudaMalloc(&memory, size) == cudaSuccess ? memory : nullptr;
  }
  bool deallocate(void* memory) noexcept override {
    return cudaFree(memory) == cudaSuccess;
  }
  std::mutex mutex;
  std::condition_variable released;
  std::atomic<bool> armed{true};
  std::atomic<bool> entered{false};
  bool proceed = false;
};

TEST_F(SharedEnginesTest, AFailedRacingLoadSharesTheEnginePublishedWhileItWaited) {
  auto* const backend_runtime = shared_runtime_for_testing();
  ASSERT_NE(backend_runtime, nullptr);
  GatingAllocator allocator;
  backend_runtime->setGpuAllocator(&allocator);
  struct ResetAllocator {
    nvinfer1::IRuntime* runtime;
    ~ResetAllocator() {
      runtime->setGpuAllocator(nullptr);
    }
  } reset{backend_runtime};
  Handle loser;
  Handle winner;
  Error loser_result = Error::Internal;
  std::thread failing([&] {
    cudaSetDevice(0);
    fail_this_load = true;
    loser_result = loser.load(blob_);
    fail_this_load = false;
  });
  const bool entered = wait_for_flag(allocator.entered);
  auto winning = std::async(std::launch::async, [&] {
    cudaSetDevice(0);
    return winner.load(blob_);
  });
  const bool published = winning.wait_for(kRendezvousDeadline) == std::future_status::ready;
  {
    std::lock_guard<std::mutex> lock(allocator.mutex);
    allocator.proceed = true;
  }
  allocator.released.notify_all();
  failing.join();
  const Error winner_result = winning.get();
  ASSERT_TRUE(entered);
  ASSERT_TRUE(published);
  ASSERT_EQ(winner_result, Error::Ok);
  ASSERT_EQ(loser_result, Error::Ok);
  EXPECT_EQ(loser.engine(), winner.engine());
  const auto reference = winner.run(1, 12);
  ASSERT_FALSE(reference.empty());
  EXPECT_EQ(loser.run(1, 12), reference);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
