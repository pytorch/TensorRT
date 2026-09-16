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
// smaller one already allocated, two of the three things that make the per-shape
// query answer zero -- the third, a failed query, cannot be induced from inside
// this process -- and two threads submitting against one pooled buffer at once.
// It also covers the one place this branch could have moved the option-off path
// without meaning to, and where it says it did not: the error returns of an
// unpooled call.
//
// The TensorRT engine is built here rather than loaded from a .pte so the target
// carries no exported artifact, at the cost of a few seconds of builder time.
//
// COVERAGE LIMIT: every test below needs a CUDA device and a TensorRT that can
// build an engine. Without one the whole suite skips and covers nothing, so a
// green run on a host with no GPU says nothing about the pool -- and a skipped
// gtest case exits zero, which Bazel reports as a passing target. A run that is
// meant to have a device says so through TORCHTRT_EXECUTORCH_REQUIRE_CUDA, and
// then the missing device is a failure instead, which each failing case says for
// itself. Where that variable is unset, a count of the cases that skipped for that
// reason is printed at the end of the suite.
//
// That variable covers the missing device and nothing else. Cases also skip for a
// second reason: the device would not fill, its memory would not stay still, or it
// has no stream-ordered allocator. None of those is a state a test can insist on
// while sharing the device with other processes. Those skips stand whether the
// variable is set or not, so a green required-CUDA run says every case ran, not
// that every case covered what it is named for. No count is given here, because a
// skip written in a fixture helper belongs to every case that calls it and moves
// the moment one is added; the suite counts these separately from the missing
// device and lists them by name and reason at the end of the suite, whether or not
// the variable is set. That list is in the test log, which --test_output=errors
// prints nothing of for a target that passes; the CI invocation therefore also
// passes --test_summary=detailed, which names every skipped case in Bazel's own
// summary.
//
// SHARED-DEVICE WARNING: the three cases that need the device full take it to
// essentially zero free bytes for as long as they hold their DeviceMemoryHog --
// measured, 16 MiB free of 81151 MiB, with a neighbour process getting
// out-of-memory on 256 MiB allocations throughout the window. That is the only way
// to reach the pool's allocation-failure path from inside this process: nothing in
// the delegate takes an allocator a test could substitute, and the other early
// returns on that path need TensorRT or CUDA to fail a call that is correct as
// made. Bazel's exclusive tag keeps other actions in the same build off this
// device; it cannot keep anything else off it.
//
// The CI invocation in .github/workflows/executorch-test-linux.yml passes that
// variable, and the job it sits in asks for a GPU runner and starts its container
// with every GPU attached. The ExecuTorch job's own gate in ci-linux-x86_64.yml
// does not keep these cases off an ordinary pull-request push: it sits out only
// when the lane is skip or the backend is RTX, and _decide.yml resolves an
// unlabelled pull_request to lane=fast and backend=standard. What does drop the
// job is the standard channel having been cancelled, which the same workflow
// already carries a comment about.

#include "torch_tensorrt/executorch/PooledScratchInstall.h"
#include "torch_tensorrt/executorch/SharedScratchPool.h"
#include "torch_tensorrt/executorch/SharedScratchPoolTestHooks.h"
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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
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

// Spelled out rather than taken from TensorRTBackend.h: a test that reads the
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

// A dynamic-batch engine whose profile admits an empty batch. Binding one is a
// valid call that needs no activation scratch, from an engine that needs some,
// which is one of the three things a zero from updateDeviceMemorySizeForShapes
// can mean. The dimensions are small because no test compares this engine's
// memory against a figure; only whether each answer it gives is zero matters.
constexpr int kDynRows = 64;
constexpr int kDynCols = 64;
constexpr int kDynMaxBatch = 128;
// Inside the same profile and not empty, so one engine covers both answers.
constexpr int kDynBatch = 4;

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

// Wraps a serialized engine in the delegate's blob wire format.
std::vector<std::uint8_t> wrap_engine_plan(const nvinfer1::IHostMemory& plan) {
  const std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],)"
      R"("hardware_compatible":false,"device_id":0})";
  const auto metadata_offset = static_cast<std::uint32_t>(kHeaderSize);
  const auto metadata_size = static_cast<std::uint32_t>(metadata.size());
  const auto engine_offset = static_cast<std::uint32_t>(align_up(metadata_offset + metadata_size, kEngineAlignment));

  std::vector<std::uint8_t> blob(static_cast<std::size_t>(engine_offset) + plan.size(), 0);
  std::memcpy(blob.data(), kMagic, sizeof(kMagic));
  write_field(blob, kMetadataOffsetField, metadata_offset);
  write_field(blob, kMetadataSizeField, metadata_size);
  write_field(blob, kEngineOffsetField, engine_offset);
  write_field(blob, kEngineSizeField, static_cast<std::uint64_t>(plan.size()));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(blob.data() + engine_offset, plan.data(), plan.size());
  return blob;
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
  return wrap_engine_plan(*plan);
}

// The scratch-needing network again, over a leading dimension the caller chooses
// per call, with an empty batch as the profile minimum.
std::vector<std::uint8_t> build_dynamic_batch_engine_blob() {
  static BuilderLogger logger;

  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (builder == nullptr) {
    return {};
  }
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
  if (network == nullptr) {
    return {};
  }

  nvinfer1::ITensor* input =
      network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims3{-1, kDynRows, kDynCols});
  if (input == nullptr || !add_scratch_needing_net(*network, *input)) {
    return {};
  }

  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (config == nullptr) {
    return {};
  }
  nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
  profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims3{0, kDynRows, kDynCols});
  profile->setDimensions("input_0", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims3{kDynBatch, kDynRows, kDynCols});
  profile->setDimensions(
      "input_0", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims3{kDynMaxBatch, kDynRows, kDynCols});
  config->addOptimizationProfile(profile);

  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (plan == nullptr) {
    return {};
  }
  return wrap_engine_plan(*plan);
}

// The activation scratch one context of `blob`'s engine needs for the given
// shape, read the way execute() reads it. Zero if the engine could not be
// measured, which is also what an empty batch answers, so a caller reading a zero
// as meaningful has to rule the failure out by some other measurement.
std::size_t measure_engine_scratch(
    const std::vector<std::uint8_t>& blob,
    int rows = kRows,
    int cols = kCols,
    int batch = 1) {
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
  if (!ctx->setInputShape("input_0", nvinfer1::Dims3{batch, rows, cols})) {
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
  // option is set to at this moment. `batch`/`rows`/`cols` must be a shape the
  // blob's profile admits; `batch` may be 0, which allocates nothing and leaves
  // both device pointers null, a state an empty ExecuTorch tensor can arrive in.
  Error load(
      const std::vector<std::uint8_t>& blob,
      std::uint32_t seed,
      int rows = kRows,
      int cols = kCols,
      int batch = 1) {
    batch_ = static_cast<SizesType>(batch);
    rows_ = static_cast<SizesType>(rows);
    cols_ = static_cast<SizesType>(cols);
    std::vector<float> host_in(elems());
    for (std::size_t i = 0; i < elems(); ++i) {
      host_in[i] = pattern(i, seed);
    }
    if (bytes() > 0) {
      if (cudaMalloc(&device_in_, bytes()) != cudaSuccess || cudaMalloc(&device_out_, bytes()) != cudaSuccess) {
        return Error::MemoryAllocationFailed;
      }
      if (cudaMemcpy(device_in_, host_in.data(), bytes(), cudaMemcpyHostToDevice) != cudaSuccess) {
        return Error::Internal;
      }
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
    if (bytes() == 0) {
      return true;
    }
    const std::vector<float> host(elems(), value);
    return cudaMemcpy(device_out_, host.data(), bytes(), cudaMemcpyHostToDevice) == cudaSuccess;
  }

  // Runs one inference on `stream`. Returns without waiting for the enqueue,
  // which is the state the pool's handoff exists to order.
  Error run(cudaStream_t stream) {
    return run_with_input(stream, device_in_);
  }

  // The same run with no CallerStreamGuard, which is one of the four things that
  // make execute() synchronize the stream before returning. It is the one a memory
  // measurement can use: the other three each need a buffer this fixture does not
  // otherwise allocate -- a host-backed input or output, which execute() stages
  // through a device buffer of its own, or an extra output EValue for an aliased
  // output -- and the growth measurement would read those alongside the pool. The
  // enqueue goes on cudaStreamPerThread, which is not ordered against any stream
  // created cudaStreamNonBlocking.
  Error run_on_the_synchronized_path() {
    return run_with_input(nullptr, device_in_, /*scope_a_caller_stream=*/false);
  }

  // The same run with the input bound to caller-owned host memory, which
  // execute() stages through a device buffer of its own instead of binding
  // directly. The output stays device-resident, so the only staging is the
  // input's. `host_in` must hold bytes() bytes.
  Error run_from_host_input(cudaStream_t stream, void* host_in) {
    return run_with_input(stream, host_in);
  }

  // The same run with the output bound to caller-owned host memory as well, which
  // execute() also stages through a device buffer of its own -- and allocates that
  // buffer on the first call that needs it -- which, with the device full, is the
  // only failure this suite can force after the input's staging copy has been
  // queued. Both pointers must hold bytes() bytes.
  Error run_from_host_input_to_host_output(cudaStream_t stream, void* host_in, void* host_out) {
    return run_with_input(stream, host_in, /*scope_a_caller_stream=*/true, host_out);
  }

  // Where execute() staged this handle's input, or null if it never had to.
  void* staging_buffer_for_input_0() const {
    const EngineHandle* const h = handle();
    return h->cached_input_ptrs.empty() ? nullptr : h->cached_input_ptrs[0];
  }

  std::vector<float> read_output() const {
    std::vector<float> host_out(elems());
    if (bytes() > 0 && cudaMemcpy(host_out.data(), device_out_, bytes(), cudaMemcpyDeviceToHost) != cudaSuccess) {
      host_out.clear();
    }
    return host_out;
  }

  const EngineHandle* handle() const {
    return static_cast<const EngineHandle*>(handle_);
  }

  // The execution context execute() configures and enqueues on, so a test can
  // watch what the delegate does to it.
  nvinfer1::IExecutionContext* context() const {
    return static_cast<EngineHandle*>(handle_)->exec_ctx.get();
  }

  std::size_t elems() const {
    return static_cast<std::size_t>(batch_) * static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_);
  }

  std::size_t bytes() const {
    return elems() * sizeof(float);
  }

 private:
  // With `scope_a_caller_stream` false no CallerStreamGuard is scoped and `stream`
  // is unread: execute() then finds no caller stream, enqueues on
  // cudaStreamPerThread and synchronizes before returning. A null `out_ptr` means
  // this handle's own device-resident output.
  Error run_with_input(cudaStream_t stream, void* in_ptr, bool scope_a_caller_stream = true, void* out_ptr = nullptr) {
    // Separate arrays: execute() resizes the output tensor to the shape TensorRT
    // inferred, which writes through whichever array that tensor was given.
    SizesType in_sizes[3] = {batch_, rows_, cols_};
    SizesType out_sizes[3] = {batch_, rows_, cols_};
    ::executorch::aten::TensorImpl in_impl(ScalarType::Float, 3, in_sizes, in_ptr);
    ::executorch::aten::TensorImpl out_impl(
        ScalarType::Float, 3, out_sizes, out_ptr != nullptr ? out_ptr : device_out_);
    ::executorch::aten::Tensor in_tensor(&in_impl);
    ::executorch::aten::Tensor out_tensor(&out_impl);
    EValue in_value(in_tensor);
    EValue out_value(out_tensor);
    EValue* args[2] = {&in_value, &out_value};

    BackendExecutionContext exec_context;
    if (!scope_a_caller_stream) {
      return backend_.execute(exec_context, handle_, Span<EValue*>(args, 2));
    }
    ::executorch::extension::cuda::CallerStreamGuard guard(stream);
    return backend_.execute(exec_context, handle_, Span<EValue*>(args, 2));
  }

  // EngineHandle is placement-newed into this arena by init(), and the arena is
  // never reset, so it only has to hold one instance.
  static constexpr std::size_t kArenaBytes = 4096;

  TensorRTBackend backend_;
  std::vector<std::uint8_t> arena_storage_;
  std::unique_ptr<MemoryAllocator> arena_;
  DelegateHandle* handle_ = nullptr;
  void* device_in_ = nullptr;
  void* device_out_ = nullptr;
  SizesType batch_ = 1;
  SizesType rows_ = kRows;
  SizesType cols_ = kCols;
};

// Sets `out` to the device-wide bytes in use, or returns false leaving it alone.
// It reports the failure rather than substituting a figure because both callers
// subtract two of these: a zero for the first reading of a pair makes the second
// look like the whole cost of what was measured between them, which is a pass.
bool device_bytes_in_use(std::size_t& out) {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    return false;
  }
  out = total_bytes - free_bytes;
  return true;
}

// Takes device memory until an allocation of the size a caller names can no longer
// succeed, and gives it all back when it goes out of scope. The only way to reach
// an allocation failure inside execute() -- the pool's, or a staging buffer's --
// from inside the process, and the reason it is held for as short a window as
// possible: while it is up, every other process on this device is out of memory
// too.
class DeviceMemoryHog {
 public:
  DeviceMemoryHog() = default;
  DeviceMemoryHog(const DeviceMemoryHog&) = delete;
  DeviceMemoryHog& operator=(const DeviceMemoryHog&) = delete;

  ~DeviceMemoryHog() {
    release();
  }

  // Chunked from large to small, so a device with tens of gigabytes free is filled
  // in a few dozen allocations rather than thousands.
  bool leave_less_free_than(std::size_t bytes) {
    for (std::size_t chunk = std::size_t{1} << 30; chunk >= (std::size_t{1} << 20); chunk /= 4) {
      while (free_bytes_above(bytes)) {
        void* block = nullptr;
        if (cudaMalloc(&block, chunk) != cudaSuccess) {
          cudaGetLastError();
          break;
        }
        blocks_.push_back(block);
      }
    }
    return !free_bytes_above(bytes);
  }

  void release() {
    for (void* block : blocks_) {
      cudaFree(block);
    }
    blocks_.clear();
  }

 private:
  static bool free_bytes_above(std::size_t bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    return cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes > bytes;
  }

  std::vector<void*> blocks_;
};

// Opens a live kUSER_MANAGED context over `blob`'s engine with one shape bound,
// the state execute() installs a scratch buffer into. Held by the caller, unlike
// measure_engine_scratch's, which is gone by the time it returns its figure.
struct LiveContext {
  TRTUniquePtr<nvinfer1::IRuntime> runtime;
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  TRTUniquePtr<nvinfer1::IExecutionContext> ctx;
};

bool open_user_managed_context(const std::vector<std::uint8_t>& blob, int rows, int cols, int batch, LiveContext& out) {
  static BuilderLogger logger;
  TensorRTBlobHeader header;
  if (!TensorRTBlobHeader::parse(blob.data(), blob.size(), header)) {
    return false;
  }
  out.runtime.reset(nvinfer1::createInferRuntime(logger));
  if (out.runtime == nullptr) {
    return false;
  }
  out.engine.reset(
      out.runtime->deserializeCudaEngine(TensorRTBlobHeader::engine_data(blob.data(), header), header.engine_size));
  if (out.engine == nullptr) {
    return false;
  }
  out.ctx.reset(out.engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
  if (out.ctx == nullptr) {
    return false;
  }
  return out.ctx->setInputShape("input_0", nvinfer1::Dims3{batch, rows, cols});
}

// Counts the reference-count calls TensorRT makes on a recorder as it attaches
// and detaches one, which is how a test attached to the delegate's own context
// sees install_pooled_scratch scope its recorder over the install.
//
// TensorRT documents setErrorRecorder as calling incRefCount on the recorder it
// takes and decRefCount on the one it replaces, and it does. That makes the count
// this starts at load-bearing rather than incidental: it starts at zero, so the
// reference TensorRT takes at the attach is the only one, which is the shape
// IErrorRecorder's documentation describes and TensorRT's own samples use -- a
// recorder handed to TensorRT and destroyed when the count returns to zero. A
// recorder that kept a reference of its own would never reach zero however the
// install swapped it, and `reached_zero` below would report nothing.
class CountingErrorRecorder final : public nvinfer1::IErrorRecorder {
 public:
  int32_t getNbErrors() const noexcept override {
    return 0;
  }
  nvinfer1::ErrorCode getErrorCode(int32_t) const noexcept override {
    return nvinfer1::ErrorCode::kSUCCESS;
  }
  ErrorDesc getErrorDesc(int32_t) const noexcept override {
    return "";
  }
  bool hasOverflowed() const noexcept override {
    return false;
  }
  void clear() noexcept override {}
  bool reportError(nvinfer1::ErrorCode, ErrorDesc) noexcept override {
    reported.fetch_add(1);
    return false;
  }
  RefCount incRefCount() noexcept override {
    reattached.fetch_add(1);
    return ++refs_;
  }
  RefCount decRefCount() noexcept override {
    detached.fetch_add(1);
    const RefCount remaining = --refs_;
    if (remaining <= 0) {
      reached_zero.store(true);
    }
    return remaining;
  }

  void forget() {
    reattached.store(0);
    detached.store(0);
    reported.store(0);
    reached_zero.store(false);
  }

  std::atomic<int> reattached{0};
  std::atomic<int> detached{0};
  std::atomic<int> reported{0};
  // Whether TensorRT ever left this recorder with no references. A recorder that
  // deletes itself there -- the shape the interface documents -- would be gone.
  std::atomic<bool> reached_zero{false};

 private:
  RefCount refs_ = 0;
};

// Set by a run that is supposed to have a GPU. A skip is then a failure, rather
// than a green target that covered nothing.
constexpr char kRequireCudaEnvVar[] = "TORCHTRT_EXECUTORCH_REQUIRE_CUDA";

bool cuda_device_is_required() {
  const char* const value = std::getenv(kRequireCudaEnvVar);
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
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
    dynamic_blob_ = build_dynamic_batch_engine_blob();
    if (blob_.empty() || scratch_free_blob_.empty() || big_blob_.empty() || dynamic_blob_.empty()) {
      return;
    }
    scratch_bytes_ = measure_engine_scratch(blob_);
    big_scratch_bytes_ = measure_engine_scratch(big_blob_, kBigRows, kBigCols);
    empty_batch_scratch_bytes_ = measure_engine_scratch(dynamic_blob_, kDynRows, kDynCols, 0);
    dynamic_batch_scratch_bytes_ = measure_engine_scratch(dynamic_blob_, kDynRows, kDynCols, kDynBatch);
    scratch_free_engine_bytes_ = engine_scratch_requirement(scratch_free_blob_);
    dynamic_engine_bytes_ = engine_scratch_requirement(dynamic_blob_);
  }

  static void TearDownTestSuite() {
    report_the_second_reason_skips();
    if (skipped_for_no_device_ == 0) {
      return;
    }
    // Nothing skipped when the requirement is on -- SetUp counts the missing device
    // and then fails the case. Printing here would put a skip banner under those
    // failures and tell the reader to set the very variable that produced them.
    if (cuda_device_is_required()) {
      return;
    }
    const ::testing::TestSuite* const suite = ::testing::UnitTest::GetInstance()->current_test_suite();
    std::fprintf(
        stderr,
        "[  SKIPPED ] %d of %d cases in this file: no CUDA device. Nothing here ran, so this target passing says "
        "nothing about the shared activation scratch pool. Set %s=1 on a run that is meant to have a device and a "
        "skip becomes a failure.\n",
        skipped_for_no_device_,
        suite == nullptr ? skipped_for_no_device_ : suite->total_test_count(),
        kRequireCudaEnvVar);
  }

  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      ++skipped_for_no_device_;
      this_case_skipped_for_no_device_ = true;
      if (cuda_device_is_required()) {
        FAIL() << "no CUDA device, and " << kRequireCudaEnvVar
               << " says this run must have one. Every case in this file needs a device, so without one the binary "
                  "would exit zero having covered nothing.";
      }
      GTEST_SKIP() << "no CUDA device: the shared-scratch backend path is not covered by this run";
    }
    ASSERT_FALSE(blob_.empty()) << "TensorRT could not build the fixture engine";
    ASSERT_FALSE(scratch_free_blob_.empty()) << "TensorRT could not build the scratch-free fixture engine";
    ASSERT_FALSE(big_blob_.empty()) << "TensorRT could not build the larger fixture engine";
    ASSERT_FALSE(dynamic_blob_.empty()) << "TensorRT could not build the dynamic-batch fixture engine";
    ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);
    // The pool outlives every test in this file, and most of them depend on what
    // it holds when they start: one expects a growth, one expects none, one
    // expects a first allocation. Leaving it alone makes each of those depend on
    // which cases ran before it, so the suite passes in declaration order and in
    // no other -- running the empty-input case first turns the handoff test's
    // first run into a growth, which is not the state that case is written for.
    //
    // Reported rather than waited out: a slot still in use here was left that way
    // by the case before, and this case cannot run without the reset, so it fails
    // now and says which state it found.
    ASSERT_TRUE(reset_shared_scratch_pool_for_testing())
        << "a device's pool slot still had its lock held when this case started, so an earlier case returned with a "
           "claim outstanding and the pool could not be reset";
  }

  void TearDown() override {
    record_a_second_reason_skip();
    set_shared_scratch(backend_, false);
    // Also here, so a buffer this test grew is not still resident while the next
    // one measures device-wide memory.
    ASSERT_TRUE(reset_shared_scratch_pool_for_testing())
        << "this case left a device's pool slot with its lock held, so the pool could not be reset for the next one";
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

  const std::vector<std::uint8_t>& dynamic_blob() const {
    return dynamic_blob_;
  }

  // Defined below, next to the cases that call them.
  void run_a_growth_and_bound_what_it_cost(bool synchronized_path);
  void run_a_growth_beside_parked_work(bool synchronized_path);

  // A case can also skip for a reason the device being present says nothing about
  // -- the device would not fill, its memory would not stay still, it has no
  // stream-ordered allocator. Those skips are the difference between a green
  // target and a run that covered what the cases are named for, and with
  // --test_output=errors a passing target prints not one of them. So they are
  // collected here and printed at the end whether or not the CUDA requirement is
  // armed, which is the case the missing-device banner cannot cover.
  void record_a_second_reason_skip() {
    if (this_case_skipped_for_no_device_ || !::testing::Test::IsSkipped()) {
      return;
    }
    const ::testing::TestInfo* const info = ::testing::UnitTest::GetInstance()->current_test_info();
    if (info == nullptr) {
      return;
    }
    std::string reason;
    const ::testing::TestResult* const result = info->result();
    for (int i = 0; result != nullptr && i < result->total_part_count(); ++i) {
      const ::testing::TestPartResult& part = result->GetTestPartResult(i);
      if (part.type() == ::testing::TestPartResult::kSkip) {
        reason = part.summary();
        break;
      }
    }
    // The reason a case gives usually wraps, and one line per skip is what makes
    // the report below readable.
    for (char& c : reason) {
      if (c == '\n') {
        c = ' ';
      }
    }
    while (!reason.empty() && reason.back() == ' ') {
      reason.pop_back();
    }
    if (reason.empty()) {
      reason = "no reason recorded";
    }
    second_reason_skips_.emplace_back(info->name(), reason);
  }

  static void report_the_second_reason_skips() {
    if (second_reason_skips_.empty()) {
      return;
    }
    const ::testing::TestSuite* const suite = ::testing::UnitTest::GetInstance()->current_test_suite();
    std::fprintf(
        stderr,
        "[  SKIPPED ] %zu of %d cases in this file ran on a device but did not cover what they are named for, so this "
        "target passing does not say the pool's growth, allocation-failure and queued-free paths were exercised:\n",
        second_reason_skips_.size(),
        suite == nullptr ? static_cast<int>(second_reason_skips_.size()) : suite->total_test_count());
    for (const auto& skip : second_reason_skips_) {
      std::fprintf(stderr, "[  SKIPPED ]   %s: %s\n", skip.first.c_str(), skip.second.c_str());
    }
  }

  TensorRTBackend backend_;
  static std::vector<std::uint8_t> blob_;
  static std::vector<std::uint8_t> scratch_free_blob_;
  static std::vector<std::uint8_t> big_blob_;
  static std::vector<std::uint8_t> dynamic_blob_;
  static std::size_t scratch_bytes_;
  static std::size_t big_scratch_bytes_;
  static std::size_t empty_batch_scratch_bytes_;
  static std::size_t dynamic_batch_scratch_bytes_;
  static std::int64_t scratch_free_engine_bytes_;
  static std::int64_t dynamic_engine_bytes_;
  static int skipped_for_no_device_;
  static std::vector<std::pair<std::string, std::string>> second_reason_skips_;
  bool this_case_skipped_for_no_device_ = false;
};

std::vector<std::uint8_t> SharedScratchBackendTest::blob_;
std::vector<std::uint8_t> SharedScratchBackendTest::scratch_free_blob_;
std::vector<std::uint8_t> SharedScratchBackendTest::big_blob_;
std::vector<std::uint8_t> SharedScratchBackendTest::dynamic_blob_;
std::size_t SharedScratchBackendTest::scratch_bytes_ = 0;
std::size_t SharedScratchBackendTest::big_scratch_bytes_ = 0;
std::size_t SharedScratchBackendTest::empty_batch_scratch_bytes_ = 0;
std::size_t SharedScratchBackendTest::dynamic_batch_scratch_bytes_ = 0;
std::int64_t SharedScratchBackendTest::scratch_free_engine_bytes_ = -1;
std::int64_t SharedScratchBackendTest::dynamic_engine_bytes_ = -1;
int SharedScratchBackendTest::skipped_for_no_device_ = 0;
std::vector<std::pair<std::string, std::string>> SharedScratchBackendTest::second_reason_skips_;

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

// The case above sends the bad entry on its own, where returning before the store
// and storing before the return look the same. A span carrying a good entry ahead
// of a bad one tells them apart, and it is the span a caller writes when it sets
// several options at once.
TEST_F(SharedScratchBackendTest, SetOptionRejectsASpanWithoutApplyingTheEntriesBeforeTheBadOne) {
  ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);

  BackendOption turn_on;
  std::strncpy(turn_on.key, kOptionKey, sizeof(turn_on.key) - 1);
  turn_on.value = true;
  // The same key again, so what the span asks for is unambiguous: this entry
  // cannot be read as addressed to some other backend.
  BackendOption wrong_type;
  std::strncpy(wrong_type.key, kOptionKey, sizeof(wrong_type.key) - 1);
  wrong_type.value = 1;
  BackendOption options[2] = {turn_on, wrong_type};
  BackendOptionContext context;

  EXPECT_EQ(backend_.set_option(context, Span<BackendOption>(options, 2)), Error::InvalidArgument);

  LoadedEngine engine;
  ASSERT_EQ(engine.load(blob(), 28), Error::Ok);
  EXPECT_FALSE(engine.handle()->shared_scratch)
      << "a span that was refused still turned the pool on for every engine loaded after it";
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
    std::size_t before = 0;
    ASSERT_TRUE(device_bytes_in_use(before)) << "cudaMemGetInfo failed, so this test measured nothing";
    std::vector<std::unique_ptr<LoadedEngine>> engines;
    for (int i = 0; i < kEngineCount; ++i) {
      engines.push_back(std::make_unique<LoadedEngine>());
      ASSERT_EQ(engines.back()->load(blob(), 9), Error::Ok);
      ASSERT_EQ(engines.back()->run(stream), Error::Ok);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::size_t after = 0;
    ASSERT_TRUE(device_bytes_in_use(after)) << "cudaMemGetInfo failed, so this test measured nothing";
    // The subtraction is unsigned, so a fall in device-wide usage would wrap it
    // to a number that satisfies the comparison at the end for free.
    ASSERT_GE(after, before) << "device-wide memory in use fell across the private-scratch measurement, so "
                                "something outside this test is releasing memory on this device";
    private_cost = after - before;
  }

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  std::size_t pooled_cost = 0;
  {
    std::size_t before = 0;
    ASSERT_TRUE(device_bytes_in_use(before)) << "cudaMemGetInfo failed, so this test measured nothing";
    std::vector<std::unique_ptr<LoadedEngine>> engines;
    for (int i = 0; i < kEngineCount; ++i) {
      engines.push_back(std::make_unique<LoadedEngine>());
      ASSERT_EQ(engines.back()->load(blob(), 9), Error::Ok);
      ASSERT_EQ(engines.back()->run(stream), Error::Ok);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    std::size_t after = 0;
    ASSERT_TRUE(device_bytes_in_use(after)) << "cudaMemGetInfo failed, so this test measured nothing";
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
// which nothing else in this file does, and bounds what the growth cost the
// device.
//
// The reading is taken the moment execute() returns and *before* any synchronize
// of this test's, which is what makes the upper bound mean something. A
// stream-ordered free hands the bytes back only at the next synchronize of the
// stream it was queued on, so a synchronize in between is exactly what would hide
// a disposal that queued the free and left it. Measured on an A100 with CUDA 13.0:
// cudaFreeAsync on a cudaMalloc'd pointer returns cudaSuccess and defers the free,
// and the bytes come back at that synchronize and at no point before it -- a
// stream cudaStreamQuery reports as drained still holds them. So a growth that
// queued the free and returned costs the whole new buffer here rather than the
// difference, and the upper bound below fails.
//
// The lower bound also fails if the pool were already large enough for the second
// engine, which is how this test could otherwise pass vacuously. The fixture
// empties the pool before each test, so it is the smaller engine's run below that
// establishes the size the growth has to exceed.
//
// `synchronized_path` picks which kind of call grows the pool. The disposal is the
// same either way and this is the same measurement pointed at each, which is the
// point: the bound holds on a call whose own synchronize would have covered for a
// free the pool did not finish, and on one that makes no synchronize at all.
void SharedScratchBackendTest::run_a_growth_and_bound_what_it_cost(bool synchronized_path) {
  ASSERT_GE(big_scratch_bytes_, scratch_bytes_ + kMinMeasurableScratch)
      << "the two fixture engines ask for " << scratch_bytes_ << " and " << big_scratch_bytes_
      << " bytes of activation scratch, too close for the growth to be measurable";

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  const auto run_one = [synchronized_path, stream](LoadedEngine& engine) {
    return synchronized_path ? engine.run_on_the_synchronized_path() : engine.run(stream);
  };

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
    ASSERT_EQ(run_one(small_priv), Error::Ok);
    ASSERT_EQ(run_one(big_priv), Error::Ok);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    small_expected = small_priv.read_output();
    big_expected = big_priv.read_output();
  }

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine small;
  ASSERT_EQ(small.load(blob(), 15), Error::Ok);
  ASSERT_TRUE(small.handle()->shared_scratch);
  ASSERT_EQ(run_one(small), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Loaded before the measurement starts: its weights and its I/O are not part of
  // what the growth costs.
  LoadedEngine big;
  ASSERT_EQ(big.load(big_blob(), 16, kBigRows, kBigCols), Error::Ok);
  ASSERT_TRUE(big.handle()->shared_scratch);

  std::size_t before = 0;
  ASSERT_TRUE(device_bytes_in_use(before)) << "cudaMemGetInfo failed, so this test measured nothing";
  const auto measurement_began = std::chrono::steady_clock::now();
  ASSERT_EQ(run_one(big), Error::Ok);
  // No synchronize between the growth and this reading: see the note above. The
  // allocation and the whole disposal are host calls execute() makes and returns
  // from, so what this reads is settled whether or not the enqueue has finished.
  std::size_t after = 0;
  ASSERT_TRUE(device_bytes_in_use(after)) << "cudaMemGetInfo failed, so this test measured nothing";
  // A control window of the same length as the measurement, with nothing of this
  // test's running in it. It is sampled before the synchronize below for the same
  // reason the reading above is: a synchronize releases a free the growth only
  // queued, so one taken inside this window would move `settled` away from `after`
  // and turn the failure the bounds are meant to report into a skip. On a device
  // this test has to itself, device-wide usage does not move across it; anything
  // else means another process is moving memory on the same timescale as the
  // growth, which is what makes the bounds below report a figure the pool did not
  // produce. The same length matters: two readings taken back to back would sample
  // microseconds against the growth's milliseconds and would miss almost
  // everything. It is still a sample of a different window, so it narrows the
  // misdiagnosis rather than removing it, and the upper bound names the cause as
  // well. The exclusive tag keeps other Bazel actions off this device, not other
  // processes.
  //
  // What counts as quiet is what the bounds below can absorb, rather than exact
  // equality, which threw the whole measurement away for a byte. The tighter of
  // the two bounds allows scratch_bytes_/2 of slack, so a control window that
  // moved by at most half of that cannot turn what they report into something
  // else, while anything large enough to matter still skips. The tolerance is for
  // allocator granularity and not for a busy device: on a device this suite had to
  // itself the window was measured moving 0 bytes against a tolerance of 8 MiB for
  // these fixture engines, and a neighbour allocating at the scale they do moves it
  // by far more than that.
  std::this_thread::sleep_for(std::chrono::steady_clock::now() - measurement_began);
  std::size_t settled = 0;
  ASSERT_TRUE(device_bytes_in_use(settled)) << "cudaMemGetInfo failed, so this test measured nothing";
  const std::size_t quiet_tolerance = scratch_bytes_ / 4;
  const std::size_t control_window_moved_by = settled > after ? settled - after : after - settled;
  const bool device_was_quiet = control_window_moved_by <= quiet_tolerance;
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  // Only something outside this test can make device-wide usage fall across a
  // growth: the growth allocates the larger buffer before it disposes of the
  // smaller one. It belongs with the skip below rather than being a failure of its
  // own, for the reason that skip exists -- and it has to be answered before the
  // subtraction, which is unsigned.
  const bool device_gave_memory_back = after < before;
  const std::size_t difference = big_scratch_bytes_ - scratch_bytes_;

  // The pool must still serve the smaller engine after the growth moved the
  // buffer: its context holds the address it was given on its previous call, and
  // that address has been freed. This run is also the one place the suite installs
  // a size below the pool's capacity, since that is what this engine's shapes
  // need, so a size TensorRT refuses fails here.
  ASSERT_TRUE(small.fill_output(kSentinel));
  ASSERT_EQ(run_one(small), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> small_actual = small.read_output();
  const std::vector<float> big_actual = big.read_output();

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_EQ(big_expected.size(), big_actual.size());
  ASSERT_FALSE(big_expected.empty());
  EXPECT_EQ(std::memcmp(big_expected.data(), big_actual.data(), big_expected.size() * sizeof(float)), 0)
      << "the engine that grew the pool did not produce what it produces with its own scratch";

  ASSERT_EQ(small_expected.size(), kElems);
  ASSERT_EQ(small_actual.size(), kElems);
  EXPECT_NE(small_expected[0], kSentinel) << "the reference output is the sentinel, so a skipped enqueue would pass";
  EXPECT_EQ(std::memcmp(small_expected.data(), small_actual.data(), kBytes), 0)
      << "the smaller engine stopped producing its own output once the growth moved the shared buffer";

  // Last, so the output comparisons above are made either way.
  if (device_gave_memory_back) {
    GTEST_SKIP() << "device-wide memory in use fell from " << before << " to " << after
                 << " bytes across the growth, which a growth cannot do, so another process released memory on this "
                    "device and the two bounds below would measure it rather than the growth";
  }
  if (!device_was_quiet) {
    GTEST_SKIP() << "device-wide memory in use moved from " << after << " to " << settled << " bytes -- "
                 << control_window_moved_by << " against a " << quiet_tolerance
                 << "-byte tolerance -- with nothing of this test's running in between, so another process is "
                    "allocating on this device and the two bounds below would measure it rather than the growth";
  }
  const std::size_t growth_cost = after - before;
  EXPECT_GE(growth_cost, difference / 2) << "the larger engine cost " << growth_cost << " bytes against a "
                                         << difference
                                         << "-byte difference in requirement, so the pool did not grow for it";
  EXPECT_LE(growth_cost, difference + scratch_bytes_ / 2)
      << "the larger engine cost " << growth_cost << " bytes, about the whole " << big_scratch_bytes_
      << "-byte buffer rather than the " << difference
      << "-byte difference, so the buffer it replaced was still resident when execute() returned -- a free the growth "
         "only queued, or none at all -- unless another process allocated on this device across the measurement, "
         "which the control window above samples for and cannot rule out";
}

// The path this option exists for: a caller stream with device-resident I/O, so
// execute() never synchronizes the stream. Nothing this test or the caller does
// would ever release a free the pool left queued, so the bound here is the one
// that says the pool finished it.
TEST_F(SharedScratchBackendTest, ALargerEngineGrowsThePoolAndFreesTheBufferItReplaces) {
  run_a_growth_and_bound_what_it_cost(/*synchronized_path=*/false);
}

// The other path, where execute() ends by synchronizing the stream. The bytes have
// to be back by the time the call returns here too, and the same bound says so --
// what this case adds is a growth beside a synchronize the disposal does not lean
// on, since the disposal returns the bytes itself and runs before it.
TEST_F(SharedScratchBackendTest, AGrowthOnASynchronizedCallFreesTheBufferItReplaces) {
  run_a_growth_and_bound_what_it_cost(/*synchronized_path=*/true);
}

// ---------------------------------------------------------------------------
// An engine that needs no activation scratch
// ---------------------------------------------------------------------------

// The pooled path branches on this flag, and it must separate the two fixture
// networks or the branch is only ever taken one way below.
TEST_F(SharedScratchBackendTest, EachEngineRecordsWhetherItClaimsFromThePool) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine needing;
  LoadedEngine scratch_free;
  ASSERT_EQ(needing.load(blob(), 12), Error::Ok);
  ASSERT_EQ(scratch_free.load(scratch_free_blob(), 13), Error::Ok);

  EXPECT_TRUE(needing.handle()->claims_pooled_scratch)
      << "the two-softmax network reports no activation scratch of its own, so no test here reaches the pooled path";
  EXPECT_FALSE(scratch_free.handle()->claims_pooled_scratch)
      << "the pointwise chain reports activation scratch, so it no longer covers the scratch-free case";
}

// Turning the pool on must not turn an engine that legitimately needs no
// activation scratch into a failure. Such an engine skips the pool altogether,
// which is what the capacity check below pins.
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
  EXPECT_EQ(shared_scratch_capacity_for_testing(pooled.handle()->device_id), 0u)
      << "an engine that needs no activation scratch under any shape claimed the pool anyway";
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
// An empty input to an engine that does need activation scratch
// ---------------------------------------------------------------------------

// An empty batch inside the profile is a valid call, and the per-shape query
// answers it with the same zero it gives for a failed query, from an engine whose
// own requirement is not zero. The three assertions at the top are what make this
// the empty-input case rather than either of the others: an engine that needs
// nothing would fail the second, and a measurement that could not read the engine
// at all would fail the third.
//
// What the empty call must not do is size the pool for itself. The engine's own
// profile-wide requirement, `dynamic_engine_bytes_`, is the figure a zero invites
// installing, and it stands far above the `dynamic_batch_scratch_bytes_` the next
// call actually needs; the pool never shrinks, so installing it would pin the pool
// at many times that for the rest of the process. The empty call is therefore
// checked twice, once against an empty pool and once against a pool holding a real
// requirement, and neither may leave the pool above what a non-empty call asked
// for. The capacity is read directly rather than inferred from device-wide
// memory, which cannot resolve the difference reliably.
TEST_F(SharedScratchBackendTest, AnEmptyInputRunsWithThePoolEnabled) {
  ASSERT_EQ(empty_batch_scratch_bytes_, 0u)
      << "an empty batch reports " << empty_batch_scratch_bytes_
      << " bytes of activation scratch, so this test no longer covers a call whose per-shape query answers zero";
  ASSERT_GT(dynamic_engine_bytes_, 0)
      << "the dynamic fixture engine reports no activation scratch of its own, so the zero above is the "
         "scratch-free case rather than the empty-input one";
  ASSERT_GT(dynamic_batch_scratch_bytes_, 0u) << "the same engine reports no activation scratch at batch " << kDynBatch
                                              << " either, so the zero above says nothing about the batch being empty";

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

  // The non-empty reference is taken with private scratch, so the pooled run
  // below has something to be compared against that the pool did not produce.
  LoadedEngine priv;
  ASSERT_EQ(priv.load(dynamic_blob(), 19, kDynRows, kDynCols, kDynBatch), Error::Ok);
  ASSERT_FALSE(priv.handle()->shared_scratch);
  ASSERT_TRUE(priv.fill_output(kSentinel));
  ASSERT_EQ(priv.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> expected = priv.read_output();

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine empty;
  ASSERT_EQ(empty.load(dynamic_blob(), 19, kDynRows, kDynCols, 0), Error::Ok);
  ASSERT_TRUE(empty.handle()->shared_scratch);
  const int device = empty.handle()->device_id;
  ASSERT_EQ(shared_scratch_capacity_for_testing(device), 0u)
      << "the fixture left the pool holding something, so an empty call adding nothing to it proves nothing";
  EXPECT_EQ(empty.run(stream), Error::Ok) << "the pool rejected an empty input to an engine that needs scratch";
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  // Not zero: enqueueV3 refuses this context with no device memory installed
  // whatever the bound shapes need, so the pool has to allocate something. What it
  // must not do is take the engine's word for how much.
  const std::size_t after_empty = shared_scratch_capacity_for_testing(device);
  // Pinned to the exact figure rather than bounded below the 131072 bytes the next
  // call needs, because a bound that loose is met by any minimum up to 128 KiB and
  // the point of the minimum is that it is the smallest thing enqueueV3 accepts.
  EXPECT_EQ(after_empty, 1u) << "an empty call against an empty pool took " << after_empty
                             << " bytes rather than the one-byte kMinPooledScratchBytes; a zero means no buffer was "
                                "installed, which TensorRT refuses for this engine, and anything larger means the "
                                "call sized the pool from a figure of its own";

  // The same engine on a shape that does need scratch, after the empty call, so a
  // pool the empty call left in an unusable state is not left untested.
  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(dynamic_blob(), 19, kDynRows, kDynCols, kDynBatch), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  ASSERT_TRUE(pooled.fill_output(kSentinel));
  EXPECT_EQ(pooled.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> actual = pooled.read_output();
  const std::size_t after_non_empty = shared_scratch_capacity_for_testing(device);
  EXPECT_EQ(after_non_empty, dynamic_batch_scratch_bytes_)
      << "the pool holds " << after_non_empty << " bytes after a call needing " << dynamic_batch_scratch_bytes_
      << ", so it was not sized to what that call asked for";

  // A second empty call, now that the pool holds a real requirement: it must be
  // handed that buffer rather than grow the pool to the engine's profile-wide one.
  ASSERT_EQ(empty.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(shared_scratch_capacity_for_testing(device), after_non_empty)
      << "the empty call grew the pool from " << after_non_empty << " bytes, against the " << dynamic_engine_bytes_
      << " bytes this engine reports over its whole profile";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  const std::size_t dyn_elems =
      static_cast<std::size_t>(kDynBatch) * static_cast<std::size_t>(kDynRows) * static_cast<std::size_t>(kDynCols);
  ASSERT_EQ(expected.size(), dyn_elems);
  ASSERT_EQ(actual.size(), dyn_elems);
  // Without these two the comparison would be satisfied by an execute() that
  // wrote nothing, and by a network whose output does not depend on its input.
  EXPECT_NE(expected[0], kSentinel) << "the reference output is the sentinel, so a skipped enqueue would pass";
  bool varies = false;
  for (std::size_t i = 1; i < dyn_elems && !varies; ++i) {
    varies = expected[i] != expected[0];
  }
  EXPECT_TRUE(varies) << "the reference output is constant, so the comparison proves nothing";
  EXPECT_EQ(std::memcmp(expected.data(), actual.data(), dyn_elems * sizeof(float)), 0);
}

// The install is what separates the two causes of a zero from the per-shape
// query. Both reach execute() as a zero and neither is asked to size the pool, so
// both can be handed a buffer as small as one byte; only one of them is safe with
// it. Where the bound shapes genuinely need nothing the engine expects nothing
// and the one-byte buffer is accepted. Where the query failed the engine expects what it
// always did, TensorRT refuses the install through a call that returns void, and
// the context keeps the buffer it was last given -- which a growth may already
// have freed, and enqueueV3 then reports success while the engine reads and
// writes memory the pool no longer owns.
//
// The failed query itself cannot be induced from inside this process, so what is
// pinned here is the check that catches its consequence, over the same TensorRT
// call execute() makes.
TEST_F(SharedScratchBackendTest, AnUndersizedScratchInstallIsSeenAsRefused) {
  LiveContext needs_scratch;
  ASSERT_TRUE(open_user_managed_context(blob(), kRows, kCols, 1, needs_scratch))
      << "could not open a user-managed context over the fixture engine";
  const std::size_t need = needs_scratch.ctx->updateDeviceMemorySizeForShapes();
  ASSERT_GT(need, 1u) << "the fixture engine needs no more than one byte of activation scratch for this shape, so an "
                         "undersized install cannot be built out of it";

  void* enough = nullptr;
  ASSERT_EQ(cudaMalloc(&enough, need), cudaSuccess);
  EXPECT_TRUE(install_pooled_scratch(*needs_scratch.ctx, enough, need, 0))
      << "a buffer of exactly what the bound shapes need was reported as refused, which would fail every pooled call";

  void* one_byte = nullptr;
  ASSERT_EQ(cudaMalloc(&one_byte, 1), cudaSuccess);
  EXPECT_FALSE(install_pooled_scratch(*needs_scratch.ctx, one_byte, 1, 0))
      << "TensorRT refused a one-byte buffer for a context expecting " << need
      << " bytes and the backend read the install as accepted, so the enqueue would run on the buffer installed "
         "before it";

  // The safe cause, which has to keep working: an empty batch expects nothing, so
  // the one-byte buffer is accepted and no refusal may be invented for it.
  LiveContext empty_batch;
  ASSERT_TRUE(open_user_managed_context(dynamic_blob(), kDynRows, kDynCols, 0, empty_batch));
  ASSERT_EQ(empty_batch.ctx->updateDeviceMemorySizeForShapes(), 0u)
      << "an empty batch does not answer zero for this engine, so this half covers nothing";
  EXPECT_TRUE(install_pooled_scratch(*empty_batch.ctx, one_byte, 1, 0))
      << "the one-byte buffer an empty call is handed was reported as refused, which would fail a call that is safe";

  EXPECT_EQ(cudaFree(enough), cudaSuccess);
  EXPECT_EQ(cudaFree(one_byte), cudaSuccess);
}

// The recorder the install swaps out is the caller's, and swapping it must not
// destroy it. TensorRT drops a reference on the recorder it replaces and takes
// one again on the restore, so a recorder whose only reference is TensorRT's
// reaches zero in between -- and destroying an error recorder once its count
// reaches zero is what IErrorRecorder's documentation describes and what
// TensorRT's samples do. Such a recorder is freed by the install and
// re-registered dangling by the restore, once per pooled call, with nothing
// reported by either TensorRT or the delegate.
//
// Driven over install_pooled_scratch rather than through execute() because what
// has to hold is a property of the swap rather than of the call around it, and
// every pooled call makes exactly this one swap.
TEST_F(SharedScratchBackendTest, AnInstallKeepsAReferenceOnTheRecorderItSwapsOut) {
  // Declared before the context so it outlives it: the context drops its
  // reference on this recorder as it is destroyed.
  CountingErrorRecorder recorder;
  LiveContext live;
  ASSERT_TRUE(open_user_managed_context(blob(), kRows, kCols, 1, live))
      << "could not open a user-managed context over the fixture engine";
  const std::size_t need = live.ctx->updateDeviceMemorySizeForShapes();
  ASSERT_GT(need, 0u) << "the fixture engine needs no activation scratch for this shape, so there is no install to "
                         "swap a recorder around";

  void* buffer = nullptr;
  ASSERT_EQ(cudaMalloc(&buffer, need), cudaSuccess);

  live.ctx->setErrorRecorder(&recorder);
  ASSERT_EQ(recorder.reattached.load(), 1)
      << "TensorRT did not take a reference on the recorder it was handed, so this recorder does not stand in for one "
         "whose only reference is TensorRT's and nothing below means anything";
  recorder.forget();

  EXPECT_TRUE(install_pooled_scratch(*live.ctx, buffer, need, 0))
      << "a buffer of exactly what the bound shapes need was reported as refused";

  ASSERT_GT(recorder.detached.load(), 0)
      << "the install never displaced the recorder attached to the context, so it made no swap for this case to watch";
  EXPECT_FALSE(recorder.reached_zero.load())
      << "the install left the caller's recorder with no references before putting it back, so a recorder of the shape "
         "TensorRT documents deletes itself at the install and the restore then re-registers freed memory -- once per "
         "pooled call";

  EXPECT_EQ(cudaFree(buffer), cudaSuccess);
}

// The two cases above drive install_pooled_scratch directly, which leaves
// execute() free to stop calling it: replacing the checked call with a bare
// setDeviceMemoryV2 installs the same buffer, produces the same output and drops
// only the refusal, so every other case here stays green.
//
// Forcing the refusal itself through execute() is not available. The size
// execute() installs is what the per-shape query just returned, over a buffer the
// pool has grown to at least that, so the installed size is never short of what
// the context expects. Only a failed query makes the two disagree, and that
// cannot be induced from inside this process.
//
// Nor is the choice between that size and the pool's capacity observable through
// TensorRT: an install larger than the context expects is accepted in silence, so
// no black-box case here can tell the two apart, and the argument for the smaller
// one is the hazard it removes rather than anything a test can read back. What the
// suite does see is the other direction -- ALargerEngineGrowsThePoolAndFreesTheBufferItReplaces
// runs the smaller engine again after the growth, so its install is smaller than
// the capacity, and a size TensorRT would not accept fails that case.
//
// What is observable is the recorder. install_pooled_scratch attaches one for
// the duration of the install and restores the previous one after, because that
// is the only channel TensorRT reports a refusal on. A recorder this test leaves
// on the delegate's context therefore sees itself replaced and put back exactly
// once per pooled run, and sees nothing at all if the install stops going
// through the helper.
TEST_F(SharedScratchBackendTest, ExecuteInstallsPooledScratchThroughTheCheckedHelper) {
  // Declared before the engine so it outlives the context TensorRT attaches it
  // to: the context decrements this recorder's count as it is destroyed.
  CountingErrorRecorder recorder;

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 25), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  ASSERT_TRUE(pooled.handle()->claims_pooled_scratch)
      << "this engine skips the pool, so its execute() makes no install for this case to watch";
  pooled.context()->setErrorRecorder(&recorder);

  recorder.forget();
  ASSERT_EQ(pooled.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  // Two of each and not one: TensorRT drops a reference on the recorder it
  // replaces and takes it again on the restore, and the install holds one of its
  // own across the swap so that the caller's recorder cannot reach zero in
  // between -- AnInstallKeepsAReferenceOnTheRecorderItSwapsOut is that half.
  EXPECT_EQ(recorder.detached.load(), 2)
      << "a pooled execute() did not replace the context's error recorder, so its activation-scratch install was not "
         "the checked one and a refusal by TensorRT would go unread";
  EXPECT_EQ(recorder.reattached.load(), 2)
      << "the recorder the caller had attached was not put back after the install, so TensorRT's diagnostics for the "
         "rest of the call go somewhere the caller did not ask for";
  EXPECT_EQ(pooled.context()->getErrorRecorder(), &recorder)
      << "execute() left an error recorder of its own attached to the context";
  EXPECT_EQ(recorder.reported.load(), 0)
      << "TensorRT reported an error to the caller's recorder during a run that succeeded";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Every pooled path out of execute() that is not the happy one returns while the
// claim still holds the device's lock, and leaves the claim's destructor to drop
// it. Nothing else here takes one of those paths, and the failure a regression
// there produces is not a wrong answer: it is every later pooled call on the
// device blocking forever with nothing logged.
//
// Only one of them can be reached from inside the process. A refused install needs
// the pool to hold less than the context expects, which it never does; a failed
// enqueue and a failed completion record need TensorRT or CUDA to fail a call that
// is correct as made. The pool's own allocation can be made to fail by taking the
// device's memory first, and it returns through the same destructor as the others.
//
// What it reaches is the lock the destructor drops, not the buffer it disposes
// of: an allocation that failed retired nothing, so this case leaves the
// destructor with none to free. Nothing in this suite reaches a bail-out that
// still holds a retired buffer, since the two returns that can are the two named
// above. What keeps that path right is the shape of release() rather than a case:
// it takes no argument, and the disposal is the same three steps on every path --
// wait on the host for the enqueue the handoff event names, cudaFreeAsync on the
// pool's stream, synchronize that stream. It never asks what kind of call it is,
// so a bail-out gets the same disposal as a happy return and there is no promise
// recorded at the claim for a later return to leave stale.
TEST_F(SharedScratchBackendTest, AFailedPooledAllocationLeavesTheDeviceLockFreeAndNothingPending) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 31), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  ASSERT_TRUE(pooled.handle()->claims_pooled_scratch) << "this engine skips the pool, so it never allocates from it";
  const int device_id = pooled.handle()->device_id;
  ASSERT_EQ(shared_scratch_capacity_for_testing(device_id), 0u)
      << "the pool already holds a buffer, so this call would reuse it rather than allocate";

  Error failed_run = Error::Ok;
  cudaError_t pending_after_the_failure = cudaSuccess;
  {
    DeviceMemoryHog hog;
    if (!hog.leave_less_free_than(scratch_bytes_)) {
      ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
      GTEST_SKIP() << "could not take the device below the " << scratch_bytes_
                   << " bytes this engine's scratch needs, so its allocation would have succeeded";
    }
    // The hog clears the last failed allocation of its own, and nothing else has
    // made a CUDA call since, so whatever is pending after the run below was left
    // by the run.
    ASSERT_EQ(cudaGetLastError(), cudaSuccess) << "something before the call under test left a CUDA error pending";
    failed_run = pooled.run(stream);
    pending_after_the_failure = cudaPeekAtLastError();
  }

  if (failed_run == Error::Ok) {
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    GTEST_SKIP() << "the pooled allocation succeeded with the device full, so this run did not take the path under "
                    "test";
  }
  ASSERT_EQ(failed_run, Error::MemoryAllocationFailed)
      << "the run failed somewhere other than the pool's allocation, so it says nothing about that path";

  SharedScratchDevice& dev = scratch_pool().get(device_id);
  const bool lock_free = dev.mu.try_lock();
  if (lock_free) {
    dev.mu.unlock();
  }
  ASSERT_TRUE(lock_free) << "a pooled call that returned early left the device's pool lock held, so every later pooled "
                            "call on this device blocks forever";

  // The failure is reported by the return value, so the CUDA error the pool's own
  // cudaMalloc left has to be cleared where it was made. Left pending it is
  // collected by whichever caller makes the next CUDA call on this thread and
  // reported under that call's name -- which is what the capture refusal has its
  // own case for, from the other direction.
  EXPECT_EQ(pending_after_the_failure, cudaSuccess)
      << "the pooled call left '" << cudaGetErrorString(pending_after_the_failure)
      << "' pending on this thread after reporting the failure through its return value, so the next CUDA call any "
         "caller makes collects the pool's error as its own";
  cudaGetLastError();

  // And the slot is still usable, not merely unlocked: with the memory back, the
  // next call allocates and runs.
  ASSERT_TRUE(pooled.fill_output(kSentinel));
  EXPECT_EQ(pooled.run(stream), Error::Ok) << "the pool was left unusable by a call whose allocation failed";
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::vector<float> actual = pooled.read_output();
  ASSERT_EQ(actual.size(), kElems);
  EXPECT_NE(actual[0], kSentinel) << "the engine did not write its output, so the run above did not reach it";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// ---------------------------------------------------------------------------
// Stream capture
// ---------------------------------------------------------------------------

// The refusal clears the CUDA error state only when the capture query itself
// failed, because then the error is the query's own. On the ordinary refusal --
// the query succeeded and reported a capture -- whatever is pending was left by
// earlier work on this thread, and clearing it takes it away from the caller,
// which learns of it from no later call either: the error is not sticky.
TEST_F(SharedScratchBackendTest, ARefusedPooledCallLeavesTheCallersPendingCudaErrorAlone) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 32), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);

  // An error of the caller's own, unread. Nothing but cudaGetLastError clears one,
  // so it is still there when the refused call returns unless that call took it.
  void* impossible = nullptr;
  ASSERT_EQ(cudaMalloc(&impossible, static_cast<std::size_t>(1) << 62), cudaErrorMemoryAllocation);

  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed), cudaSuccess);
  const Error refused = pooled.run(stream);
  cudaGraph_t graph = nullptr;
  const cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  // Read here, before any assertion can return early with it still pending and
  // hand it to the next case.
  const cudaError_t pending = cudaGetLastError();

  ASSERT_EQ(refused, Error::NotSupported) << "the run was not refused, so it did not take the path under test";
  EXPECT_EQ(end_err, cudaSuccess) << "the refused run invalidated the capture: " << cudaGetErrorString(end_err);
  EXPECT_EQ(pending, cudaErrorMemoryAllocation)
      << "the refusal cleared an error this backend did not cause, so the caller never sees it: got "
      << cudaGetErrorString(pending);

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// That execute() refuses a capturing stream, and where in execute() it does it.
// Under the default Global mode the host wait on the previous enqueue, and further
// down the cudaMalloc that grows a host-input staging buffer, are prohibited and
// invalidate the capture where they stand. A refusal after either is a clean error
// handed to a caller whose capture is already dead, which is the outcome the guard
// exists to prevent. Relaxed mode could not tell the two apart: it permits
// everything execute() does before it reaches the pool, so a guard sitting
// anywhere ahead of the pool would look the same.
//
// The first run is what arms that wait -- it returns with its enqueue still in
// flight, and the next call waits for it before touching the context. It also
// leaves the pool holding a buffer with an enqueue recorded against it, so the
// handoff wait the refusal protects is one this call would really have made.
TEST_F(SharedScratchBackendTest, APooledEngineRefusesACaptureBeforeAnythingCanInvalidateIt) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  // A node of the caller's own, so a null graph below means the capture was
  // invalidated rather than that nothing was ever captured.
  void* captured_target = nullptr;
  ASSERT_EQ(cudaMalloc(&captured_target, 16), cudaSuccess);

  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 21), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  ASSERT_EQ(pooled.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_TRUE(pooled.handle()->inflight_pending)
      << "the first run left no enqueue in flight, so the next call makes no host wait and this case pins nothing";
  ASSERT_GT(shared_scratch_capacity_for_testing(pooled.handle()->device_id), 0u)
      << "the pool holds nothing, so the run below has no handoff to wait on and the capture it is refused for was "
         "never at risk";

  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(captured_target, 0, 16, stream), cudaSuccess);
  const Error captured = pooled.run(stream);
  cudaGraph_t graph = nullptr;
  const cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
  const bool have_graph = graph != nullptr;
  if (have_graph) {
    cudaGraphDestroy(graph);
  }

  EXPECT_EQ(captured, Error::NotSupported)
      << "execute() did not refuse a pooled run on a stream capturing in the default mode";
  EXPECT_EQ(end_err, cudaSuccess) << "the refused run had already made a call the capture could not take: "
                                  << cudaGetErrorString(end_err);
  EXPECT_TRUE(have_graph) << "the capture ended with no graph, so the run invalidated it before refusing";

  EXPECT_EQ(cudaFree(captured_target), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
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

// Long enough to tell a wait on the stream carrying it from no wait at all, short
// enough not to lengthen the suite.
constexpr std::chrono::milliseconds kHostWorkOnTheCallerStream{1500};

// Host work that runs on its own and needs no releasing, unlike hold_stream: what
// it is for is measuring whether something waited for the stream carrying it, in
// a place where a gate nothing opened would deadlock the teardown instead.
void CUDART_CB sleep_on_the_stream(void*) {
  std::this_thread::sleep_for(kHostWorkOnTheCallerStream);
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

// A thread that is joined however the case leaves its scope, so an assertion that
// returns early does not leave one joinable and take the process down with it.
class JoinAtScopeExit {
 public:
  explicit JoinAtScopeExit(std::thread t) : t_(std::move(t)) {}
  JoinAtScopeExit(const JoinAtScopeExit&) = delete;
  JoinAtScopeExit& operator=(const JoinAtScopeExit&) = delete;

  ~JoinAtScopeExit() {
    join();
  }

  void join() {
    if (t_.joinable()) {
      t_.join();
    }
  }

 private:
  std::thread t_;
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
// What a growth waits for
// ---------------------------------------------------------------------------

// A growth's disposal of the buffer it replaces waits for one thing and not the
// other, and these three cases park each of them in turn.
//
// It waits for the enqueue against the buffer it retires, because freeing under a
// live enqueue is what the marker event is there to stop; the third case pins that
// it waits rather than freeing under it. It does not wait for anything else the
// device is running, because the free is queued on a stream the pool owns rather
// than made device-wide; the first two pin that, one on each kind of call. Both
// kinds, because what kind of call it is must not decide which disposal it gets.
//
// Each measures what the call came back before through the gate's watchdog, which
// is the only other thing that can open a gate: if a call waited when it should
// not have, the watchdog would have had to open the gate to end the test, and the
// case says so.

// Nothing the growing engine or the pool ever submitted to runs on the parked
// stream, so only a device-wide free has any reason to wait for it.
//
// Gated on the device's stream-ordered allocator, and gated before the gate is
// parked so a device without one costs milliseconds rather than the watchdog
// interval. Where there is none, cudaFreeAsync reports cudaErrorNotSupported, the
// backend correctly falls back to a device-wide free, and this assertion would
// report a defect that is not there -- the target is built for sbsa as well as
// x86_64, and nothing in the delegate asks the device about memory pools, so the
// fallback is what a whole platform would take.
//
// On a device that does have one, nothing in this suite covers that fallback.
// Reaching it needs cudaFreeAsync to fail a call that is correct as made, or the
// pool's stream creation to fail, and neither can be induced from inside this
// process.
void SharedScratchBackendTest::run_a_growth_beside_parked_work(bool synchronized_path) {
  ASSERT_GE(big_scratch_bytes_, scratch_bytes_ + kMinMeasurableScratch)
      << "the two fixture engines ask for " << scratch_bytes_ << " and " << big_scratch_bytes_
      << " bytes of activation scratch, too close for the second to be sure of growing the pool";
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  cudaStream_t unrelated = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&unrelated, cudaStreamNonBlocking), cudaSuccess);

  // `stream` is unread on the synchronized path: with no CallerStreamGuard
  // execute() enqueues on cudaStreamPerThread and synchronizes before returning.
  const auto run_one = [synchronized_path, stream](LoadedEngine& engine) {
    return synchronized_path ? engine.run_on_the_synchronized_path() : engine.run(stream);
  };

  LoadedEngine small;
  LoadedEngine big;
  ASSERT_EQ(small.load(blob(), 26), Error::Ok);
  ASSERT_EQ(big.load(big_blob(), 27, kBigRows, kBigCols), Error::Ok);
  ASSERT_TRUE(small.handle()->shared_scratch);
  ASSERT_TRUE(big.handle()->shared_scratch);
  const int device_id = big.handle()->device_id;

  int memory_pools = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&memory_pools, cudaDevAttrMemoryPoolsSupported, device_id), cudaSuccess);
  if (memory_pools == 0) {
    GTEST_SKIP() << "device " << device_id
                 << " reports no stream-ordered allocator (cudaDevAttrMemoryPoolsSupported = 0), so a growth here "
                    "correctly falls back to a device-wide free, which does wait for work parked anywhere on the "
                    "device";
  }

  // Without this the larger engine allocates rather than grows, and a growth that
  // retires nothing disposes of nothing. Drained, so the only enqueue the growth's
  // wait can be held by is the one it submits itself, which nothing here parks;
  // the parked stream is the only other thing that could hold the growth up.
  ASSERT_EQ(run_one(small), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::size_t before = shared_scratch_capacity_for_testing(device_id);
  ASSERT_GT(before, 0u) << "the smaller engine left the pool empty, so the larger one has nothing to retire";

  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(unrelated, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, unrelated);

  ASSERT_TRUE(big.fill_output(kSentinel));
  const Error growth_error = run_one(big);
  const bool came_back_with_the_device_held = !gate.forced_open.load();

  gate_release.release();
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  EXPECT_EQ(growth_error, Error::Ok);
  EXPECT_GT(shared_scratch_capacity_for_testing(device_id), before)
      << "the pool did not grow, so no buffer was retired and nothing was disposed of";
  {
    // The slot has the pool's own stream for this device, which only a growth
    // with a buffer to dispose of ever asks for. What this does not say is which
    // stream the free was queued on: the stream is created in the claim, so it is
    // here whatever the disposal went on to do with it.
    // AGrowthsDisposalDoesNotSynchronizeTheCallersStream below is what pins that.
    SharedScratchDevice& dev = scratch_pool().get(device_id);
    std::lock_guard<std::mutex> lock(dev.mu);
    EXPECT_NE(dev.disposal_stream, nullptr)
        << "the growth disposed of the buffer it retired without the pool's own stream for this device";
  }
  const std::vector<float> grown_output = big.read_output();
  ASSERT_FALSE(grown_output.empty());
  EXPECT_NE(grown_output[0], kSentinel)
      << "the growing engine did not write its output, so it never reached the engine";

  EXPECT_TRUE(came_back_with_the_device_held)
      << "the growing call did not return until the watchdog released a host function parked on a stream it never "
         "submitted to, so its disposal of the retired buffer waits for the whole device";

  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(unrelated), cudaSuccess);
}

// With no caller stream, execute() ends by synchronizing its own stream.
TEST_F(SharedScratchBackendTest, AGrowthDoesNotWaitForUnrelatedWorkQueuedOnTheDevice) {
  run_a_growth_beside_parked_work(/*synchronized_path=*/true);
}

// The path this option exists for, and the one a disposal that read the caller
// could not serve: nothing on it ever synchronizes the caller's stream, so a free
// queued there would never come back and the only alternative to the pool's own
// stream is a device-wide free. This case is that difference -- the growth here
// returns while a host function is still parked on an unrelated stream, which a
// device-wide free waits out.
TEST_F(SharedScratchBackendTest, AGrowthOnACallerStreamDoesNotWaitForUnrelatedWorkQueuedOnTheDevice) {
  run_a_growth_beside_parked_work(/*synchronized_path=*/false);
}

// The wait the disposal does make. Freeing the retired buffer while an enqueue is
// still reading it corrupts that engine's output and reports nothing, so the
// disposal waits on the marker event first, and what has to hold is that it waits
// rather than freeing under the enqueue.
//
// The smaller engine's own enqueue is parked behind a host function, and a thread
// releases it a short while after the growing call starts. A growth that waited
// comes back after that release; one that freed the retired buffer under the live
// enqueue comes back before it.
//
// The smaller engine's output is checked against the same engine run with private
// scratch, because it is the one whose buffer was retired underneath it: a free
// that landed early would take its activations with it.
TEST_F(SharedScratchBackendTest, AGrowthOnACallerStreamWaitsForTheEnqueueOnTheBufferItRetires) {
  ASSERT_GE(big_scratch_bytes_, scratch_bytes_ + kMinMeasurableScratch)
      << "the two fixture engines ask for " << scratch_bytes_ << " and " << big_scratch_bytes_
      << " bytes of activation scratch, too close for the second to be sure of growing the pool";

  // Long enough that the growing call cannot come back after it by accident, short
  // enough not to lengthen the suite.
  constexpr std::chrono::milliseconds kHoldTheEnqueueFor{500};

  cudaStream_t held = nullptr;
  cudaStream_t growing = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&held, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&growing, cudaStreamNonBlocking), cudaSuccess);

  // Loaded with the option off, so this one keeps its own scratch and its output
  // is what the pooled run below has to reproduce.
  LoadedEngine reference;
  ASSERT_EQ(set_shared_scratch(backend_, false), Error::Ok);
  ASSERT_EQ(reference.load(blob(), 29), Error::Ok);
  ASSERT_FALSE(reference.handle()->shared_scratch);
  ASSERT_EQ(reference.run(held), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(held), cudaSuccess);
  const std::vector<float> expected = reference.read_output();
  ASSERT_EQ(expected.size(), kElems);

  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);
  LoadedEngine small;
  LoadedEngine big;
  ASSERT_EQ(small.load(blob(), 29), Error::Ok);
  ASSERT_EQ(big.load(big_blob(), 30, kBigRows, kBigCols), Error::Ok);
  ASSERT_TRUE(small.handle()->shared_scratch);
  ASSERT_TRUE(big.handle()->shared_scratch);
  const int device_id = big.handle()->device_id;
  ASSERT_TRUE(small.fill_output(kSentinel));

  // Parked ahead of the smaller engine's enqueue, so that enqueue and the event
  // recorded after it both stay pending for as long as this test wants them to.
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(held, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, held);

  ASSERT_EQ(small.run(held), Error::Ok);
  const std::size_t before = shared_scratch_capacity_for_testing(device_id);
  ASSERT_GT(before, 0u) << "the smaller engine left the pool empty, so the larger one has nothing to retire";

  // Opens the gate, without the stream synchronization GateRelease::release() also
  // makes: that one runs on this thread at scope exit, after the join below, so the
  // two never touch GateRelease at once.
  std::atomic<bool> the_enqueue_was_released{false};
  JoinAtScopeExit opener{std::thread([&] {
    std::this_thread::sleep_for(kHoldTheEnqueueFor);
    the_enqueue_was_released.store(true);
    {
      std::lock_guard<std::mutex> lock(gate.mu);
      gate.open = true;
    }
    gate.cv.notify_all();
  })};

  const Error growth_error = big.run(growing);
  const bool waited_for_the_parked_enqueue = the_enqueue_was_released.load();
  opener.join();

  gate_release.release();
  ASSERT_EQ(cudaStreamSynchronize(held), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(growing), cudaSuccess);

  EXPECT_EQ(growth_error, Error::Ok);
  ASSERT_FALSE(gate.forced_open.load())
      << "the watchdog had to open the gate, so the growing call blocked for the whole watchdog interval rather than "
         "for the enqueue it had to wait for";
  EXPECT_TRUE(waited_for_the_parked_enqueue)
      << "the growing call returned while the enqueue against the buffer it retired was still parked, so it freed that "
         "buffer under a live enqueue";
  EXPECT_GT(shared_scratch_capacity_for_testing(device_id), before)
      << "the pool did not grow, so no buffer was retired and there was nothing to wait for";
  const std::vector<float> small_output = small.read_output();
  ASSERT_EQ(small_output.size(), kElems);
  EXPECT_EQ(std::memcmp(expected.data(), small_output.data(), kBytes), 0)
      << "the engine whose scratch buffer the growth retired did not produce what the same engine produces with "
         "private scratch, so the buffer went away while its enqueue was still using it";

  ASSERT_EQ(cudaStreamDestroy(held), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(growing), cudaSuccess);
}

// None of the growth cases above can see whether the disposal is made with the
// device's lock held: each runs its growth alone, so a lock held across it holds
// nothing up and they all stay green with the unlock moved below the disposal.
// This one is what covers that.
//
// It matters because the disposal waits on the host for the enqueue against the
// buffer it retires, which is a whole inference. Under the lock, that wait is one
// every other pooled engine on the device has to sit through before it can even
// submit, which is the serialization the execute() contract says a pooled call
// does not impose past its own submission.
//
// So: park a host function ahead of the growing engine's own enqueue, which is
// what that wait is for, then check the pool lock while the growth is still inside
// it. The capacity read under that same try_lock is what stops the case passing on
// a lock that is free because the growth has not started -- past a growth it is
// above what the smaller engine left, and the growth has not returned.
TEST_F(SharedScratchBackendTest, AGrowthOnACallerStreamDisposesWithTheDeviceLockDropped) {
  ASSERT_GE(big_scratch_bytes_, scratch_bytes_ + kMinMeasurableScratch)
      << "the two fixture engines ask for " << scratch_bytes_ << " and " << big_scratch_bytes_
      << " bytes of activation scratch, too close for the second to be sure of growing the pool";
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine small;
  LoadedEngine big;
  ASSERT_EQ(small.load(blob(), 32), Error::Ok);
  ASSERT_EQ(big.load(big_blob(), 33, kBigRows, kBigCols), Error::Ok);
  ASSERT_TRUE(small.handle()->shared_scratch);
  ASSERT_TRUE(big.handle()->shared_scratch);
  const int device_id = big.handle()->device_id;

  // Without this the larger engine allocates rather than grows, and a growth that
  // retires nothing frees nothing. Drained, so the only enqueue the growth's wait
  // can be held by is the one it submits itself, below the gate.
  ASSERT_EQ(small.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::size_t before = shared_scratch_capacity_for_testing(device_id);
  ASSERT_GT(before, 0u) << "the smaller engine left the pool empty, so the larger one has nothing to retire";

  // Ahead of the growing engine's enqueue, so that enqueue and the marker recorded
  // after it stay pending and the disposal's wait for them does too.
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(stream, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, stream);

  std::atomic<bool> growth_returned{false};
  Error growth_error = Error::Internal;
  JoinAtScopeExit grower{std::thread([&] {
    growth_error = big.run(stream);
    growth_returned.store(true);
  })};

  SharedScratchDevice& dev = scratch_pool().get(device_id);
  bool lock_free_during_the_disposal = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!growth_returned.load() && std::chrono::steady_clock::now() < deadline) {
    if (dev.mu.try_lock()) {
      const std::size_t capacity_now = dev.capacity;
      dev.mu.unlock();
      if (capacity_now > before && !growth_returned.load()) {
        lock_free_during_the_disposal = true;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const bool still_inside_execute = !growth_returned.load();

  gate_release.release();
  grower.join();
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_FALSE(gate.forced_open.load())
      << "the watchdog had to open the gate rather than this test, so more than the watchdog interval passed between "
         "parking the host function ahead of the growing engine's enqueue and releasing it, and nothing here was "
         "measured under the conditions it describes";
  ASSERT_EQ(growth_error, Error::Ok);
  ASSERT_TRUE(still_inside_execute)
      << "the growing call returned before the gate opened, so its disposal never stalled and there was no window in "
         "which to observe the lock";
  ASSERT_GT(shared_scratch_capacity_for_testing(device_id), before)
      << "the pool did not grow, so no buffer was retired and nothing was disposed of";
  EXPECT_TRUE(lock_free_during_the_disposal)
      << "the device's pool lock stayed held for the whole of a stalled growth disposal, so every other pooled engine "
         "on this device waits out an inference it has nothing to do with before it can submit";
}

// Which stream the disposal queues its free on and synchronizes, which none of
// the cases above can see. The pool's stream carries that free and nothing else,
// so synchronizing it waits for the free. The caller's carries whatever the caller
// put there, so synchronizing that waits for work this delegate never submitted --
// the same hazard the staged-input drain's contract names, arriving by another
// route. Every other case in this file stays green with the free and its
// synchronize moved onto the caller's stream.
//
// What tells the two apart is work queued on the caller's stream *after* the
// growing call recorded its own enqueue, because that is the only work on that
// stream the disposal's wait for the retired buffer's enqueue does not already
// cover -- the wait is on the handoff marker, and the growing call has recorded
// itself there. So: park the growing engine's enqueue behind a gate, wait until
// the slot counts its disposal, which is raised after that enqueue and its
// marker record are both queued, queue host work on the same stream behind them,
// and then let the enqueue go. A disposal on the pool's stream comes back while
// that host work is still running. One on the caller's stream waits it out.
TEST_F(SharedScratchBackendTest, AGrowthsDisposalDoesNotSynchronizeTheCallersStream) {
  ASSERT_GE(big_scratch_bytes_, scratch_bytes_ + kMinMeasurableScratch)
      << "the two fixture engines ask for " << scratch_bytes_ << " and " << big_scratch_bytes_
      << " bytes of activation scratch, too close for the second to be sure of growing the pool";
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine small;
  LoadedEngine big;
  ASSERT_EQ(small.load(blob(), 34), Error::Ok);
  ASSERT_EQ(big.load(big_blob(), 35, kBigRows, kBigCols), Error::Ok);
  ASSERT_TRUE(small.handle()->shared_scratch);
  ASSERT_TRUE(big.handle()->shared_scratch);
  const int device_id = big.handle()->device_id;

  int memory_pools = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&memory_pools, cudaDevAttrMemoryPoolsSupported, device_id), cudaSuccess);
  if (memory_pools == 0) {
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    GTEST_SKIP() << "device " << device_id
                 << " reports no stream-ordered allocator (cudaDevAttrMemoryPoolsSupported = 0), so this growth "
                    "correctly falls back to a device-wide free, which waits for the caller's stream whichever stream "
                    "it was offered";
  }

  // Without this the larger engine allocates rather than grows, and a growth that
  // retires nothing disposes of nothing. Drained, so the only enqueue the growth's
  // wait can be held by is the one it submits itself, below the gate.
  ASSERT_EQ(small.run(stream), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  const std::size_t before = shared_scratch_capacity_for_testing(device_id);
  ASSERT_GT(before, 0u) << "the smaller engine left the pool empty, so the larger one has nothing to retire";

  // Ahead of the growing engine's enqueue, so its disposal stalls on the wait for
  // that enqueue and this thread gets a window to queue behind it.
  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(stream, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, stream);

  std::atomic<bool> growth_returned{false};
  Error growth_error = Error::Internal;
  JoinAtScopeExit grower{std::thread([&] {
    growth_error = big.run(stream);
    growth_returned.store(true);
  })};

  // A capacity above what the smaller engine left, read under the device's lock
  // while the growing call has not returned, means the growth is past its
  // allocation and inside the disposal: the release that disposes is the last
  // thing between the two.
  SharedScratchDevice& dev = scratch_pool().get(device_id);
  bool the_disposal_started = false;
  const auto disposal_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!growth_returned.load() && std::chrono::steady_clock::now() < disposal_deadline) {
    if (dev.mu.try_lock()) {
      const std::size_t capacity_now = dev.capacity;
      dev.mu.unlock();
      if (capacity_now > before && !growth_returned.load()) {
        the_disposal_started = true;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  // Queued whether or not the wait above saw the disposal, so that a case that
  // missed the window leaves the stream in the same shape for the teardown below
  // and fails on the assertion rather than on where the work landed.
  const cudaError_t queued_behind = cudaLaunchHostFunc(stream, sleep_on_the_stream, nullptr);

  // Opened here rather than through gate_release, whose release() also
  // synchronizes the stream -- which is the very wait being measured.
  {
    std::lock_guard<std::mutex> lock(gate.mu);
    gate.open = true;
  }
  gate.cv.notify_all();
  const auto let_the_enqueue_go = std::chrono::steady_clock::now();

  const auto give_up_at = let_the_enqueue_go + kHostWorkOnTheCallerStream + std::chrono::seconds(5);
  while (!growth_returned.load() && std::chrono::steady_clock::now() < give_up_at) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const auto came_back_after = std::chrono::steady_clock::now() - let_the_enqueue_go;

  grower.join();
  gate_release.release();
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_EQ(queued_behind, cudaSuccess) << "could not queue the caller's own work behind the growing enqueue";
  ASSERT_FALSE(gate.forced_open.load())
      << "the watchdog had to open the gate rather than this test, so more than the watchdog interval passed while the "
         "growing engine's enqueue was parked and nothing here was measured under the conditions it describes";
  ASSERT_EQ(growth_error, Error::Ok);
  ASSERT_TRUE(the_disposal_started)
      << "the growing call's disposal was never seen in flight, so the caller's work may have been queued ahead of the "
         "enqueue rather than behind it and this case measured nothing";
  ASSERT_GT(shared_scratch_capacity_for_testing(device_id), before)
      << "the pool did not grow, so no buffer was retired and nothing was disposed of";
  EXPECT_LT(came_back_after, kHostWorkOnTheCallerStream / 3)
      << "the growing call came back " << std::chrono::duration_cast<std::chrono::milliseconds>(came_back_after).count()
      << " ms after its own enqueue was let go, with " << kHostWorkOnTheCallerStream.count()
      << " ms of the caller's own work queued on that stream behind it, so its disposal synchronized the caller's "
         "stream rather than the pool's -- which makes every growth wait for whatever the caller has queued";
}

// ---------------------------------------------------------------------------
// A failing call and the host-input copy it queued
// ---------------------------------------------------------------------------

// An input that is not device-resident is staged with cudaMemcpyAsync from the
// caller's own memory, and from pinned memory that copy reads the caller's bytes
// when the stream reaches it, not when it is queued. So a call that fails after
// queueing one must not return while it is still pending: the caller owns that
// buffer again the moment execute() returns, and what it writes there is what
// the device then receives.
//
// The failure driven here is the pool's allocation, the one early return in the
// pooled scratch block that can be reached from inside this process. The stream
// is parked ahead of the copy so it cannot complete on its own, and the caller
// overwrites its buffer as soon as the call comes back -- which is exactly what a
// caller may do.
//
// Three outcomes are separated at the end, by giving the failing call a
// different input from the successful one before it: the sentinel means the copy
// took the caller's post-return write, the first call's pattern means it never
// ran at all, and the second call's pattern is the only pass. Which of the three
// happens is decided by where the caller's write falls relative to the copy and
// not by any interval this case picks, so a slow return cannot turn the first
// outcome into the third.
TEST_F(SharedScratchBackendTest, AFailedPooledCallDrainsTheHostInputCopyItQueued) {
  ASSERT_EQ(set_shared_scratch(backend_, true), Error::Ok);

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  LoadedEngine pooled;
  ASSERT_EQ(pooled.load(blob(), 41), Error::Ok);
  ASSERT_TRUE(pooled.handle()->shared_scratch);
  const int device_id = pooled.handle()->device_id;

  // Pinned: cudaMemcpyAsync from pageable memory does not return until the
  // driver has taken the bytes, so a later write cannot reach the device and
  // there is no hazard to pin.
  float* host_in = nullptr;
  ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&host_in), pooled.bytes(), cudaHostAllocDefault), cudaSuccess);
  for (std::size_t i = 0; i < pooled.elems(); ++i) {
    host_in[i] = pattern(i, 41);
  }

  // One good run first, so the failing one reaches the copy: the staging buffer
  // is allocated on the call that first needs it, and with the device full that
  // allocation would fail ahead of everything this case is about.
  ASSERT_EQ(pooled.run_from_host_input(stream, host_in), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  void* const staging = pooled.staging_buffer_for_input_0();
  ASSERT_NE(staging, nullptr) << "the run bound the caller's host memory directly, so nothing was staged and there is "
                                 "no asynchronous copy to leave running";

  // The second call's input, so what the device holds at the end says which of
  // the three outcomes happened.
  for (std::size_t i = 0; i < pooled.elems(); ++i) {
    host_in[i] = pattern(i, 42);
  }

  // Empties the pool, so the failing run allocates instead of reusing what the
  // run above left in it.
  ASSERT_TRUE(reset_shared_scratch_pool_for_testing());
  ASSERT_EQ(shared_scratch_capacity_for_testing(device_id), 0u);

  DeviceMemoryHog hog;
  if (!hog.leave_less_free_than(scratch_bytes_)) {
    ASSERT_EQ(cudaFreeHost(host_in), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    GTEST_SKIP() << "could not take the device below the " << scratch_bytes_
                 << " bytes this engine's scratch needs, so its allocation would have succeeded";
  }

  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(stream, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, stream);

  // The caller overwrites its own buffer the moment the call comes back, in the
  // thread that made the call, which is what a caller is entitled to do and is
  // what decides this case. A build that returns with the copy still queued does
  // that write before the gate opens, so the copy reads the sentinel and the
  // comparison at the end sees it. A build that drains cannot reach the write
  // until the copy has run. Neither outcome depends on how long anything takes:
  // the wait below is a bound on this case, not the thing that discriminates.
  std::atomic<bool> call_returned{false};
  std::atomic<bool> caller_overwrote_its_buffer{false};
  Error failed_run = Error::Ok;
  std::thread caller([&] {
    failed_run = pooled.run_from_host_input(stream, host_in);
    call_returned.store(true);
    for (std::size_t i = 0; i < pooled.elems(); ++i) {
      host_in[i] = kSentinel;
    }
    caller_overwrote_its_buffer.store(true);
  });

  // Waited for rather than slept through, so a build that returns early is not
  // handed the rest of the interval to finish its write in. Reaching the deadline
  // is the passing outcome: the stream cannot move until the gate opens, so a call
  // that waits for its own copy is still inside execute() here.
  const auto give_up_waiting_at = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (!caller_overwrote_its_buffer.load() && std::chrono::steady_clock::now() < give_up_waiting_at) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool returned_with_the_copy_still_queued = call_returned.load();
  gate_release.release();
  caller.join();
  // After the gate, because a free is device-wide and would otherwise block on it.
  hog.release();
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

  std::vector<float> staged(pooled.elems(), 0.0f);
  ASSERT_EQ(cudaMemcpy(staged.data(), staging, pooled.bytes(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFreeHost(host_in), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_FALSE(gate.forced_open.load())
      << "the watchdog had to open the gate, so the timings below were not measured under the conditions they describe";
  if (failed_run == Error::Ok) {
    GTEST_SKIP() << "the pooled allocation succeeded with the device full, so this run did not fail where the case "
                    "needs it to";
  }
  ASSERT_EQ(failed_run, Error::MemoryAllocationFailed)
      << "the run failed somewhere other than the pool's allocation, so it says nothing about that early return";

  EXPECT_FALSE(returned_with_the_copy_still_queued)
      << "the failing call returned while the copy reading the caller's host input was still queued behind a parked "
         "stream, so whatever the caller writes next is what the device receives";
  std::size_t elements_the_device_did_not_get = 0;
  for (std::size_t i = 0; i < pooled.elems(); ++i) {
    if (staged[i] != pattern(i, 42)) {
      ++elements_the_device_did_not_get;
    }
  }
  EXPECT_EQ(elements_the_device_did_not_get, 0u)
      << "the device does not hold the input this call was made with. It holds " << staged[0]
      << " at element 0: " << kSentinel << " is what the caller wrote after execute() returned, " << pattern(0, 41)
      << " is the previous call's input and means the copy never ran, and " << pattern(0, 42) << " is a pass";
}

// The drain above belongs to the pooled path, and this branch's account of
// itself rests on it staying there: with the option off an engine is meant to be
// what it was before the option existed, down to what its error returns wait for.
// A guard that read the staging flag alone would put a whole-stream wait on every
// error return an unpooled call makes after that copy -- the rest of the input
// loop, inferShapes, the whole output-binding loop, enqueueV3 and the returns
// past it -- and that wait covers work the caller queued before calling, which
// this delegate never submitted and cannot bound.
//
// The failure driven here is the allocation of the device buffer an output that
// is not device-resident is staged through. It is made after the input's copy has
// been queued, and it is the only failure past that point this suite can force on
// an unpooled call. The stream is parked ahead of the copy, so a call that drains
// cannot return until this case opens the gate, and one that does not returns
// straight away.
//
// This is not a claim that the copy outliving the return is harmless on this
// path. It is what this delegate did before this branch, and closing it means
// changing what a default-off call does; what this case pins is that this branch
// did not change it by accident.
TEST_F(SharedScratchBackendTest, AFailedUnpooledCallLeavesTheHostInputCopyItQueuedAlone) {
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

  // Loaded with the option off, which the fixture set before this case began.
  LoadedEngine unpooled;
  ASSERT_EQ(unpooled.load(blob(), 43), Error::Ok);
  ASSERT_FALSE(unpooled.handle()->shared_scratch)
      << "the engine took the pooled path, so this case says nothing about the option-off one";

  // Pinned, for the same reason as the case above: a copy from pageable memory
  // waits for the parked stream itself, so the call would block whatever the
  // guard does and nothing here would discriminate.
  float* host_in = nullptr;
  ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&host_in), unpooled.bytes(), cudaHostAllocDefault), cudaSuccess);
  for (std::size_t i = 0; i < unpooled.elems(); ++i) {
    host_in[i] = pattern(i, 43);
  }
  std::vector<float> host_out(unpooled.elems(), 0.0f);

  // One good run first, so the failing one reaches the copy rather than failing
  // on the input's own staging allocation with the device full. Its output stays
  // device-resident, so the output staging buffer is still unallocated -- which is
  // what the failing run below fails on.
  ASSERT_EQ(unpooled.run_from_host_input(stream, host_in), Error::Ok);
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_NE(unpooled.staging_buffer_for_input_0(), nullptr)
      << "the run bound the caller's host memory directly, so nothing was staged and there is no asynchronous copy to "
         "leave running";

  DeviceMemoryHog hog;
  if (!hog.leave_less_free_than(unpooled.bytes())) {
    ASSERT_EQ(cudaFreeHost(host_in), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    GTEST_SKIP() << "could not take the device below the " << unpooled.bytes()
                 << " bytes this call's output staging needs, so its allocation would have succeeded";
  }

  StreamGate gate;
  ASSERT_EQ(cudaLaunchHostFunc(stream, hold_stream, &gate), cudaSuccess);
  GateRelease gate_release(gate, stream);

  std::atomic<bool> call_returned{false};
  Error failed_run = Error::Ok;
  std::thread caller([&] {
    failed_run = unpooled.run_from_host_input_to_host_output(stream, host_in, host_out.data());
    call_returned.store(true);
  });

  // Generous, because reaching it is the failing outcome and nothing here is
  // timed: a call that returns is waited for and no longer, and a call that
  // drains cannot return before the gate opens however long this waits.
  const auto give_up_waiting_at = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (!call_returned.load() && std::chrono::steady_clock::now() < give_up_waiting_at) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool returned_while_the_stream_was_parked = call_returned.load();

  gate_release.release();
  caller.join();
  // After the gate, because a free is device-wide and would otherwise block on it.
  hog.release();
  ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  ASSERT_EQ(cudaFreeHost(host_in), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

  ASSERT_FALSE(gate.forced_open.load())
      << "the watchdog had to open the gate rather than this case, so what is asserted below was not observed under "
         "the conditions it describes";
  if (failed_run == Error::Ok) {
    GTEST_SKIP() << "the output staging allocation succeeded with the device full, so this run did not fail where the "
                    "case needs it to";
  }
  ASSERT_EQ(failed_run, Error::MemoryAllocationFailed)
      << "the run failed somewhere other than the output staging allocation, so it says nothing about the error "
         "returns between a staged input's copy and the synchronize on the success path";

  EXPECT_TRUE(returned_while_the_stream_was_parked)
      << "a failing call with the option off waited on the caller's stream, which nothing before this branch made it "
         "do. The option is documented as costing an unpooled call a bool test or two per execute() and nothing else, "
         "and this is a wait with no bound: the caller's stream carries work this delegate never submitted";
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
  // below; releasing both threads together caught nearly all of them.
  //
  // The wait has a deadline because the loop it sits in contains a delegate call,
  // and a pooled call that blocks is one of the failures this case exists to
  // catch. Without one the thread whose partner is stuck spins here until the
  // target's own timeout, which reports as a killed binary rather than as this
  // case; with one, the run ends and the assertion below names it.
  //
  // The loop below stops at the first rendezvous that expires, which is what makes
  // the deadline affordable. It is armed per rendezvous -- a whole-case wall clock
  // would have to cover 60 real runs and would fail a slow machine instead -- and
  // paid at most once per thread, so the worst case is a bounded 30 s rather than
  // the 60 x 30 s a loop that carried on would cost against this target's 900 s
  // timeout. Stopping also keeps the surviving thread out of the delegate call
  // that follows the rendezvous, which its wedged partner may be holding the
  // device's pool lock across.
  constexpr std::chrono::seconds kRendezvousDeadline{30};
  std::atomic<int> arrived{0};
  std::atomic<bool> partner_never_arrived{false};
  auto submit_together = [&](int iteration) {
    arrived.fetch_add(1);
    const auto deadline = std::chrono::steady_clock::now() + kRendezvousDeadline;
    while (arrived.load() < 2 * (iteration + 1)) {
      if (partner_never_arrived.load() || std::chrono::steady_clock::now() >= deadline) {
        partner_never_arrived.store(true);
        return;
      }
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
      if (partner_never_arrived.load()) {
        break;
      }
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

  EXPECT_FALSE(partner_never_arrived.load())
      << "a thread waited out the " << kRendezvousDeadline.count()
      << "-second rendezvous deadline without its partner arriving, so a pooled call blocked instead of returning and "
         "both threads stopped short of "
      << kConcurrentRunsPerThread << " runs";
  EXPECT_EQ(failures.load(), 0) << "a run failed outright, so fewer than " << (2 * kConcurrentRunsPerThread)
                                << " runs reached the comparison below";
  EXPECT_EQ(wrong_outputs.load(), 0) << wrong_outputs.load() << " of " << (2 * kConcurrentRunsPerThread)
                                     << " concurrent pooled runs did not produce what the same engine produces with "
                                        "its own scratch";
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
