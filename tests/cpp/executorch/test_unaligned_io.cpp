/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Device buffers whose address is not on the 256-byte boundary TensorRT requires,
// such as a tensor viewed at an offset into a larger allocation. Bound directly,
// the engine either returns a wrong answer or faults and leaves the CUDA context
// unusable, which is why this is a target of its own: a fault here must not fail
// the other suites.

#include "torch_tensorrt/executorch/EngineHandle.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using namespace ::executorch::aten;
using namespace ::executorch::runtime;

constexpr int32_t kRows = 1024;
constexpr int32_t kCols = 1024;
constexpr size_t kElements = static_cast<size_t>(kRows) * kCols;
constexpr size_t kBytes = kElements * sizeof(float);
// One float past the boundary, the offset of x[1:] on a float32 tensor.
constexpr size_t kUnaligned = sizeof(float);

// y = 2x + 1. Through a matrix multiply by 2I it reaches kernels that misbehave on an unaligned
// address. The pointwise form is the one an aliased output can run in place; its engine also maps
// a small second input to a second output, so a call can elide the aliased output and still have
// one of its own.
std::vector<uint8_t> build_blob(bool alias) {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder ? builder->createNetworkV2(0) : nullptr);
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder ? builder->createBuilderConfig() : nullptr);
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{kRows, kCols});
  static const float scale = 2.0f;
  static const float bias = 1.0f;
  static const std::vector<float> doubling = [] {
    std::vector<float> w(static_cast<size_t>(kCols) * kCols, 0.0f);
    for (int32_t i = 0; i < kCols; ++i) {
      w[static_cast<size_t>(i) * kCols + i] = scale;
    }
    return w;
  }();
  nvinfer1::ITensor* doubled = nullptr;
  if (alias) {
    auto* multiplier = network->addConstant({2, {1, 1}}, {nvinfer1::DataType::kFLOAT, &scale, 1});
    doubled =
        network->addElementWise(*input, *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  } else {
    auto* weights = network->addConstant(
        nvinfer1::Dims2{kCols, kCols},
        {nvinfer1::DataType::kFLOAT, doubling.data(), static_cast<int64_t>(doubling.size())});
    doubled =
        network
            ->addMatrixMultiply(
                *input, nvinfer1::MatrixOperation::kNONE, *weights->getOutput(0), nvinfer1::MatrixOperation::kNONE)
            ->getOutput(0);
  }
  auto* addend = network->addConstant({2, {1, 1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
  auto* out =
      network->addElementWise(*doubled, *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  out->setName("output_0");
  network->markOutput(*out);
  if (alias) {
    auto* side = network->addInput("input_1", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, 1});
    auto* side_out =
        network->addElementWise(*side, *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
    side_out->setName("output_1");
    network->markOutput(*side_out);
  }
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    return {};
  }
  const std::string metadata = alias
      ? R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"input_1","is_input":true},)"
        R"({"name":"output_0","is_input":false},{"name":"output_1","is_input":false}],"device_id":0,)"
        R"("aliased_io":[{"output":"output_0","input":"input_0","kind":"user"}]})"
      : R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0})";
  const uint32_t metadata_offset = 32;
  const auto metadata_size = static_cast<uint32_t>(metadata.size());
  const uint32_t engine_offset = (metadata_offset + metadata_size + 15) / 16 * 16;
  const uint64_t engine_size = plan->size();
  std::vector<uint8_t> blob(engine_offset + engine_size, 0);
  std::memcpy(blob.data(), alias ? "TR02" : "TR01", 4);
  std::memcpy(blob.data() + 4, &metadata_offset, sizeof(metadata_offset));
  std::memcpy(blob.data() + 8, &metadata_size, sizeof(metadata_size));
  std::memcpy(blob.data() + 12, &engine_offset, sizeof(engine_offset));
  std::memcpy(blob.data() + 16, &engine_size, sizeof(engine_size));
  std::memcpy(blob.data() + metadata_offset, metadata.data(), metadata.size());
  std::memcpy(blob.data() + engine_offset, plan->data(), plan->size());
  return blob;
}

class UnalignedIoTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      return;
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    blob_ = build_blob(false);
    alias_blob_ = build_blob(true);
  }

  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      const char* required = std::getenv("TORCHTRT_EXECUTORCH_REQUIRE_CUDA");
      if (required && std::strcmp(required, "1") == 0) {
        FAIL() << "This run requires CUDA; no unaligned I/O test could execute";
      }
      GTEST_SKIP() << "No CUDA device; unaligned I/O is not covered";
    }
    ASSERT_FALSE(blob_.empty());
    ASSERT_FALSE(alias_blob_.empty());
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // Room for the largest offset past either end of the tensor.
    ASSERT_EQ(cudaMalloc(&input_base_, kBytes + 256), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_base_, kBytes + 256), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&side_, 512), cudaSuccess);
    host_.resize(kElements);
    for (size_t i = 0; i < kElements; ++i) {
      host_[i] = static_cast<float>(i % 97) / 8.0f;
    }
  }

  void TearDown() override {
    // SetUp skipped or stopped before taking any device memory.
    if (input_base_ == nullptr) {
      return;
    }
    if (handle_ != nullptr) {
      backend_.destroy(handle_);
    }
    cudaFree(input_base_);
    cudaFree(output_base_);
    cudaFree(side_);
    // A fault inside the engine is sticky, so the device has to be usable after every case.
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  }

  void load(const std::vector<uint8_t>& blob) {
    BackendInitContext context(&allocator_, nullptr, nullptr, nullptr);
    FreeableBuffer processed(blob.data(), blob.size(), nullptr);
    auto result = backend_.init(context, &processed, {});
    ASSERT_TRUE(result.ok());
    handle_ = static_cast<EngineHandle*>(result.get());
  }

  float* input_at(size_t offset) {
    auto* input = reinterpret_cast<float*>(static_cast<uint8_t*>(input_base_) + offset);
    EXPECT_EQ(cudaMemcpy(input, host_.data(), kBytes, cudaMemcpyHostToDevice), cudaSuccess);
    return input;
  }

  float* output_at(size_t offset) {
    auto* output = reinterpret_cast<float*>(static_cast<uint8_t*>(output_base_) + offset);
    EXPECT_EQ(cudaMemset(output, 0xff, kBytes), cudaSuccess);
    return output;
  }

  Error execute(float* input, float* output) {
    SizesType sizes[] = {kRows, kCols};
    TensorImpl in(ScalarType::Float, 2, sizes, input);
    TensorImpl out(ScalarType::Float, 2, sizes, output);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    return backend_.execute(context, handle_, {args, 2});
  }

  // Elides the aliased output when `aliased_slot` is null, which is how a zero-copy KV program calls.
  Error execute_aliased(float* buffer, float* aliased_slot) {
    SizesType sizes[] = {kRows, kCols};
    SizesType side_sizes[] = {1, 1};
    TensorImpl in(ScalarType::Float, 2, sizes, buffer);
    TensorImpl side_in(ScalarType::Float, 2, side_sizes, side_);
    TensorImpl slot(ScalarType::Float, 2, sizes, aliased_slot);
    TensorImpl side_out(ScalarType::Float, 2, side_sizes, static_cast<uint8_t*>(side_) + 256);
    EValue in_value{Tensor(&in)};
    EValue side_in_value{Tensor(&side_in)};
    EValue slot_value{Tensor(&slot)};
    EValue side_out_value{Tensor(&side_out)};
    BackendExecutionContext context;
    if (aliased_slot == nullptr) {
      EValue* args[] = {&in_value, &side_in_value, &side_out_value};
      return backend_.execute(context, handle_, {args, 3});
    }
    EValue* args[] = {&in_value, &side_in_value, &slot_value, &side_out_value};
    return backend_.execute(context, handle_, {args, 4});
  }

  // The engine applied `times` times to the staged host values.
  void expect_applied(const float* device, int times) {
    std::vector<float> actual(kElements);
    ASSERT_EQ(cudaMemcpy(actual.data(), device, kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < kElements; ++i) {
      float expected = host_[i];
      for (int t = 0; t < times; ++t) {
        expected = expected * 2.0f + 1.0f;
      }
      ASSERT_FLOAT_EQ(actual[i], expected) << "element " << i;
    }
  }

  static std::vector<uint8_t> blob_;
  static std::vector<uint8_t> alias_blob_;
  TensorRTBackend backend_;
  alignas(EngineHandle) uint8_t arena_[4096]{};
  MemoryAllocator allocator_{sizeof(arena_), arena_};
  EngineHandle* handle_ = nullptr;
  void* input_base_ = nullptr;
  void* output_base_ = nullptr;
  void* side_ = nullptr;
  std::vector<float> host_;
};

std::vector<uint8_t> UnalignedIoTest::blob_;
std::vector<uint8_t> UnalignedIoTest::alias_blob_;

TEST_F(UnalignedIoTest, AlignedBuffersAreBoundWithoutAStagingCopy) {
  load(blob_);
  float* output = output_at(0);
  ASSERT_EQ(execute(input_at(0), output), Error::Ok);
  expect_applied(output, 1);
  EXPECT_EQ(handle_->cached_input_ptrs[0], nullptr);
  EXPECT_EQ(handle_->cached_output_ptrs[0], nullptr);
}

TEST_F(UnalignedIoTest, AnUnalignedInputIsStagedAndGivesTheRightAnswer) {
  load(blob_);
  float* output = output_at(0);
  ASSERT_EQ(execute(input_at(kUnaligned), output), Error::Ok);
  expect_applied(output, 1);
  EXPECT_NE(handle_->cached_input_ptrs[0], nullptr);
}

TEST_F(UnalignedIoTest, AnUnalignedOutputIsStagedAndGivesTheRightAnswer) {
  load(blob_);
  float* output = output_at(kUnaligned);
  ASSERT_EQ(execute(input_at(0), output), Error::Ok);
  expect_applied(output, 1);
  EXPECT_NE(handle_->cached_output_ptrs[0], nullptr);
}

TEST_F(UnalignedIoTest, AnUnalignedAliasedInputStillReceivesTheInPlaceUpdate) {
  load(alias_blob_);
  float* buffer = input_at(kUnaligned);
  float* slot = output_at(0);
  ASSERT_EQ(execute_aliased(buffer, slot), Error::Ok);
  expect_applied(buffer, 1);
  expect_applied(slot, 1);
  // The second call reads what the first wrote back, so a copy-back that went nowhere fails here.
  ASSERT_EQ(execute_aliased(buffer, slot), Error::Ok);
  expect_applied(buffer, 2);
  expect_applied(slot, 2);
}

TEST_F(UnalignedIoTest, AnUnalignedAliasedInputReceivesTheUpdateWhenItsOutputIsElided) {
  load(alias_blob_);
  float* buffer = input_at(kUnaligned);
  ASSERT_EQ(execute_aliased(buffer, nullptr), Error::Ok);
  expect_applied(buffer, 1);
  ASSERT_EQ(execute_aliased(buffer, nullptr), Error::Ok);
  expect_applied(buffer, 2);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
