/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// A tensor whose type differs from its engine binding would be read or written as the binding's type. A float64 model
// compiled without truncate_double gives such a program: its engine binds float32 while the program keeps float64.

#include "torch_tensorrt/executorch/EngineHandle.h"

#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

using namespace ::executorch::aten;
using namespace ::executorch::runtime;

constexpr int32_t kElements = 4;
// Room for the widest type a test passes.
constexpr size_t kBufferBytes = kElements * sizeof(double);

// The delegate's blob: a header, the binding metadata, then the serialized engine. With `alias`, the program
// declares output_0 a user alias of input_0.
std::vector<uint8_t> wrap_plan(const nvinfer1::IHostMemory& plan, bool alias = false) {
  std::string metadata =
      R"({"io_bindings":[{"name":"input_0","is_input":true},{"name":"output_0","is_input":false}],"device_id":0)";
  if (alias) {
    metadata += R"(,"aliased_io":[{"output":"output_0","input":"input_0","kind":"user"}])";
  }
  metadata += "}";
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

// y = 2x + 1 on float32.
std::vector<uint8_t> build_blob() {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder ? builder->createNetworkV2(0) : nullptr);
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder ? builder->createBuilderConfig() : nullptr);
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {kElements}});
  static const float scale = 2.0f;
  static const float bias = 1.0f;
  auto* multiplier = network->addConstant({1, {1}}, {nvinfer1::DataType::kFLOAT, &scale, 1});
  auto* addend = network->addConstant({1, {1}}, {nvinfer1::DataType::kFLOAT, &bias, 1});
  auto* doubled =
      network->addElementWise(*input, *multiplier->getOutput(0), nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto* out =
      network->addElementWise(*doubled, *addend->getOutput(0), nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  out->setName("output_0");
  network->markOutput(*out);
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  return plan ? wrap_plan(*plan) : std::vector<uint8_t>{};
}

// Passes its input of type `type` through as `output_type`.
std::vector<uint8_t> build_cast_blob(nvinfer1::DataType type, nvinfer1::DataType output_type, bool alias = false) {
  static TRTLogger logger;
  TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  const auto strongly_typed = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
  TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder ? builder->createNetworkV2(strongly_typed) : nullptr);
  TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder ? builder->createBuilderConfig() : nullptr);
  if (!network || !config) {
    return {};
  }
  auto* input = network->addInput("input_0", type, nvinfer1::Dims{1, {kElements}});
  nvinfer1::ILayer* layer = nullptr;
  if (input) {
    layer = output_type == type ? static_cast<nvinfer1::ILayer*>(network->addIdentity(*input))
                                : network->addCast(*input, output_type);
  }
  if (!layer) {
    return {};
  }
  layer->getOutput(0)->setName("output_0");
  network->markOutput(*layer->getOutput(0));
  TRTUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  return plan ? wrap_plan(*plan, alias) : std::vector<uint8_t>{};
}

class BindingTypesTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    runtime_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      return;
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    blob_ = build_blob();
  }

  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      const char* required = std::getenv("TORCHTRT_EXECUTORCH_REQUIRE_CUDA");
      if (required && std::strcmp(required, "1") == 0) {
        FAIL() << "This run requires CUDA; no binding type test could execute";
      }
      GTEST_SKIP() << "No CUDA device; binding types are not covered";
    }
    ASSERT_FALSE(blob_.empty());
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&input_, kBufferBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&output_, kBufferBytes), cudaSuccess);
    const std::vector<float> ones(kElements, 1.0f);
    ASSERT_EQ(cudaMemcpy(input_, ones.data(), kElements * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(output_, 0, kBufferBytes), cudaSuccess);
    BackendInitContext context(&allocator_, nullptr, nullptr, nullptr);
    FreeableBuffer processed(blob_.data(), blob_.size(), nullptr);
    auto result = backend_.init(context, &processed, {});
    ASSERT_TRUE(result.ok());
    handle_ = result.get();
  }

  void TearDown() override {
    if (handle_ != nullptr) {
      backend_.destroy(handle_);
    }
    cudaFree(input_);
    cudaFree(output_);
  }

  Error execute(ScalarType input_type, ScalarType output_type) {
    SizesType sizes[] = {kElements};
    DimOrderType dim_order[] = {0};
    StridesType strides[] = {1};
    TensorImpl in(input_type, 1, sizes, input_, dim_order, strides);
    TensorImpl out(output_type, 1, sizes, output_, dim_order, strides);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    const Error err = backend_.execute(context, handle_, {args, 2});
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return err;
  }

  std::vector<float> output() {
    std::vector<float> values(kElements);
    EXPECT_EQ(cudaMemcpy(values.data(), output_, kElements * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
    return values;
  }

  static std::vector<uint8_t> blob_;
  TensorRTBackend backend_;
  alignas(EngineHandle) uint8_t arena_[4096]{};
  MemoryAllocator allocator_{sizeof(arena_), arena_};
  DelegateHandle* handle_ = nullptr;
  void* input_ = nullptr;
  void* output_ = nullptr;
};

std::vector<uint8_t> BindingTypesTest::blob_;

TEST_F(BindingTypesTest, AnInputOfAnotherTypeIsRefusedBeforeTheEngineRuns) {
  EXPECT_EQ(execute(ScalarType::Double, ScalarType::Float), Error::InvalidProgram);
  // Same width as the binding's float32, so the check is on the type and not the size.
  EXPECT_EQ(execute(ScalarType::Int, ScalarType::Float), Error::InvalidProgram);
  EXPECT_EQ(output(), std::vector<float>(kElements, 0.0f));
  // A refused call leaves the handle usable.
  ASSERT_EQ(execute(ScalarType::Float, ScalarType::Float), Error::Ok);
  EXPECT_EQ(output(), std::vector<float>(kElements, 3.0f));
  // Also checked once the binding cache holds this input's address and shape.
  EXPECT_EQ(execute(ScalarType::Int, ScalarType::Float), Error::InvalidProgram);
}

// TensorRT's INT4, FP4 and E8M0 types have no ExecuTorch type, so a binding of one of them accepts no tensor.
TEST_F(BindingTypesTest, ABindingWithNoExecuTorchTypeRefusesEveryTensor) {
  std::vector<uint8_t> blob;
  for (const auto type : {nvinfer1::DataType::kINT4, nvinfer1::DataType::kFP4, nvinfer1::DataType::kE8M0}) {
    blob = build_cast_blob(type, type);
    if (!blob.empty()) {
      break;
    }
  }
  if (blob.empty()) {
    GTEST_SKIP() << "TensorRT built no engine with an INT4, FP4 or E8M0 binding on this GPU";
  }
  alignas(EngineHandle) uint8_t arena[4096]{};
  MemoryAllocator allocator(sizeof(arena), arena);
  BackendInitContext init_context(&allocator, nullptr, nullptr, nullptr);
  FreeableBuffer processed(blob.data(), blob.size(), nullptr);
  auto handle = backend_.init(init_context, &processed, {});
  ASSERT_TRUE(handle.ok());
  EXPECT_EQ(static_cast<EngineHandle*>(handle.get())->binding_types[0], ScalarType::Undefined);
  for (const ScalarType type : {ScalarType::Byte, ScalarType::Char, ScalarType::Float}) {
    SCOPED_TRACE(::executorch::runtime::toString(type));
    SizesType sizes[] = {kElements};
    DimOrderType dim_order[] = {0};
    StridesType strides[] = {1};
    TensorImpl in(type, 1, sizes, input_, dim_order, strides);
    TensorImpl out(type, 1, sizes, output_, dim_order, strides);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    EXPECT_EQ(backend_.execute(context, handle.get(), {args, 2}), Error::InvalidProgram);
  }
  backend_.destroy(handle.get());
}

// Inputs are checked first, so a float input lets an output of a type with no ExecuTorch type be refused on its own.
TEST_F(BindingTypesTest, AnOutputWithNoExecuTorchTypeRefusesEveryTensor) {
  std::vector<uint8_t> blob;
  for (const auto type : {nvinfer1::DataType::kINT4, nvinfer1::DataType::kFP4, nvinfer1::DataType::kE8M0}) {
    blob = build_cast_blob(nvinfer1::DataType::kFLOAT, type);
    if (!blob.empty()) {
      break;
    }
  }
  if (blob.empty()) {
    GTEST_SKIP() << "TensorRT built no engine with a float input and an INT4, FP4 or E8M0 output on this GPU";
  }
  alignas(EngineHandle) uint8_t arena[4096]{};
  MemoryAllocator allocator(sizeof(arena), arena);
  BackendInitContext init_context(&allocator, nullptr, nullptr, nullptr);
  FreeableBuffer processed(blob.data(), blob.size(), nullptr);
  auto handle = backend_.init(init_context, &processed, {});
  ASSERT_TRUE(handle.ok());
  EXPECT_EQ(static_cast<EngineHandle*>(handle.get())->binding_types[1], ScalarType::Undefined);
  for (const ScalarType type : {ScalarType::Byte, ScalarType::Char, ScalarType::Float}) {
    SCOPED_TRACE(::executorch::runtime::toString(type));
    SizesType sizes[] = {kElements};
    DimOrderType dim_order[] = {0};
    StridesType strides[] = {1};
    TensorImpl in(ScalarType::Float, 1, sizes, input_, dim_order, strides);
    TensorImpl out(type, 1, sizes, output_, dim_order, strides);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    EXPECT_EQ(backend_.execute(context, handle.get(), {args, 2}), Error::InvalidProgram);
  }
  backend_.destroy(handle.get());
}

TEST_F(BindingTypesTest, AnOutputOfAnotherTypeIsRefusedBeforeTheEngineRuns) {
  EXPECT_EQ(execute(ScalarType::Float, ScalarType::Double), Error::InvalidProgram);
  EXPECT_EQ(output(), std::vector<float>(kElements, 0.0f));
}

// Each TensorRT type the check maps to an ExecuTorch type: a tensor of that type runs, and a tensor of another
// type with the same width is refused.
TEST_F(BindingTypesTest, EachMappedBindingTypeAcceptsOnlyItsOwnType) {
  struct Case {
    nvinfer1::DataType binding;
    ScalarType own;
    ScalarType other;
    size_t element_bytes;
  };
  const Case cases[] = {
      {nvinfer1::DataType::kHALF, ScalarType::Half, ScalarType::BFloat16, 2},
      {nvinfer1::DataType::kBF16, ScalarType::BFloat16, ScalarType::Half, 2},
      {nvinfer1::DataType::kINT32, ScalarType::Int, ScalarType::Float, 4},
      {nvinfer1::DataType::kINT64, ScalarType::Long, ScalarType::Double, 8},
      {nvinfer1::DataType::kBOOL, ScalarType::Bool, ScalarType::Byte, 1},
      {nvinfer1::DataType::kUINT8, ScalarType::Byte, ScalarType::Char, 1},
      {nvinfer1::DataType::kINT8, ScalarType::Char, ScalarType::Byte, 1},
      {nvinfer1::DataType::kFP8, ScalarType::Float8_e4m3fn, ScalarType::Byte, 1},
  };
  // TensorRT builds float8 layers only on GPUs with FP8 support.
  constexpr int kFirstFp8Capability = 89;
  int major = 0;
  int minor = 0;
  ASSERT_EQ(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0), cudaSuccess);
  ASSERT_EQ(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0), cudaSuccess);
  const bool has_fp8 = major * 10 + minor >= kFirstFp8Capability;
  for (const Case& c : cases) {
    if (c.binding == nvinfer1::DataType::kFP8 && !has_fp8) {
      continue;
    }
    SCOPED_TRACE(::executorch::runtime::toString(c.own));
    const std::vector<uint8_t> blob = build_cast_blob(c.binding, c.binding);
    ASSERT_FALSE(blob.empty()) << "TensorRT built no pass-through engine for this type";
    alignas(EngineHandle) uint8_t arena[4096]{};
    MemoryAllocator allocator(sizeof(arena), arena);
    BackendInitContext init_context(&allocator, nullptr, nullptr, nullptr);
    FreeableBuffer processed(blob.data(), blob.size(), nullptr);
    auto handle = backend_.init(init_context, &processed, {});
    ASSERT_TRUE(handle.ok());
    // Alternating zero and one bytes are a valid bit pattern of every one of these types.
    const size_t bytes = kElements * c.element_bytes;
    std::vector<uint8_t> pattern(bytes), actual(bytes);
    for (size_t i = 0; i < bytes; ++i) {
      pattern[i] = static_cast<uint8_t>(i % 2);
    }
    ASSERT_EQ(cudaMemcpy(input_, pattern.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
    const auto run = [&](ScalarType input_type, ScalarType output_type) {
      SizesType sizes[] = {kElements};
      DimOrderType dim_order[] = {0};
      StridesType strides[] = {1};
      TensorImpl in(input_type, 1, sizes, input_, dim_order, strides);
      TensorImpl out(output_type, 1, sizes, output_, dim_order, strides);
      EValue in_value{Tensor(&in)};
      EValue out_value{Tensor(&out)};
      EValue* args[] = {&in_value, &out_value};
      BackendExecutionContext context;
      const Error err = backend_.execute(context, handle.get(), {args, 2});
      EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
      return err;
    };
    ASSERT_EQ(cudaMemset(output_, 0xff, kBufferBytes), cudaSuccess);
    EXPECT_EQ(run(c.other, c.own), Error::InvalidProgram);
    EXPECT_EQ(run(c.own, c.other), Error::InvalidProgram);
    EXPECT_EQ(run(c.own, c.own), Error::Ok);
    ASSERT_EQ(cudaMemcpy(actual.data(), output_, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, pattern);
    backend_.destroy(handle.get());
  }
  if (!has_fp8) {
    GTEST_SKIP() << "This GPU has no FP8, so the float8 binding was not checked";
  }
}

// An aliased output is written into its input's buffer, and a program may pass no output tensor for it, so a
// pair of different types is refused at load. A tensor passed for an aliased output is checked like any other.
TEST_F(BindingTypesTest, AnAliasBetweenBindingsOfDifferentTypesIsRefusedAtLoad) {
  const auto mismatched = build_cast_blob(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kHALF, true);
  const auto matched = build_cast_blob(nvinfer1::DataType::kFLOAT, nvinfer1::DataType::kFLOAT, true);
  ASSERT_FALSE(mismatched.empty());
  ASSERT_FALSE(matched.empty());
  alignas(EngineHandle) uint8_t arena[4096]{};
  MemoryAllocator allocator(sizeof(arena), arena);
  BackendInitContext init_context(&allocator, nullptr, nullptr, nullptr);
  FreeableBuffer refused(mismatched.data(), mismatched.size(), nullptr);
  auto failed = backend_.init(init_context, &refused, {});
  ASSERT_FALSE(failed.ok());
  EXPECT_EQ(failed.error(), Error::InvalidProgram);

  FreeableBuffer accepted(matched.data(), matched.size(), nullptr);
  auto handle = backend_.init(init_context, &accepted, {});
  ASSERT_TRUE(handle.ok());
  const auto run = [&](ScalarType output_type) {
    SizesType sizes[] = {kElements};
    DimOrderType dim_order[] = {0};
    StridesType strides[] = {1};
    TensorImpl in(ScalarType::Float, 1, sizes, input_, dim_order, strides);
    TensorImpl out(output_type, 1, sizes, output_, dim_order, strides);
    EValue in_value{Tensor(&in)};
    EValue out_value{Tensor(&out)};
    EValue* args[] = {&in_value, &out_value};
    BackendExecutionContext context;
    const Error err = backend_.execute(context, handle.get(), {args, 2});
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    return err;
  };
  EXPECT_EQ(run(ScalarType::Int), Error::InvalidProgram);
  EXPECT_EQ(output(), std::vector<float>(kElements, 0.0f));
  EXPECT_EQ(run(ScalarType::Float), Error::Ok);
  EXPECT_EQ(output(), std::vector<float>(kElements, 1.0f));
  backend_.destroy(handle.get());
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
