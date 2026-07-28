/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Self-check for the optimization-profile selection policy. The policy is the
 * only non-obvious part of multi-profile support and depends on nothing but the
 * profile bounds table, so it runs here without a GPU, a TensorRT engine, or an
 * ExecuTorch method.
 */

#include "OptimizationProfileSelection.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <initializer_list>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

nvinfer1::Dims dims(std::initializer_list<int64_t> extents) {
  nvinfer1::Dims out{};
  out.nbDims = static_cast<int32_t>(extents.size());
  int i = 0;
  for (int64_t extent : extents) {
    out.d[i++] = extent;
  }
  return out;
}

InputProfileBounds bounds(std::initializer_list<int64_t> min, std::initializer_list<int64_t> max) {
  return InputProfileBounds{dims(min), dims(max)};
}

// An LLM-shaped engine: one [1, seq] input, profile 0 decodes a single token and
// profile 1 covers prefill. The ranges overlap at seq == 1 on purpose, which is
// what makes the sticky rule observable.
ProfileTable decode_and_prefill() {
  ProfileTable table;
  table.bounds = {
      {bounds({1, 1}, {1, 1})}, // profile 0: decode
      {bounds({1, 1}, {1, 2048})}, // profile 1: prefill
  };
  table.all_inputs_static = false;
  return table;
}

std::vector<nvinfer1::Dims> decode_input() {
  return {dims({1, 1})};
}

std::vector<nvinfer1::Dims> prefill_input() {
  return {dims({1, 512})};
}

TEST(ExecuTorchOptimizationProfileSelection, UnsetRequestRunsProfileZeroWhateverTheShapesSay) {
  ProfileTable table = decode_and_prefill();
  table.active = 1;
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kUnset, 0, prefill_input(), selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 0);
}

TEST(ExecuTorchOptimizationProfileSelection, PinnedRequestTakesTheIndexVerbatim) {
  ProfileTable table = decode_and_prefill();
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kPinned, 1, prefill_input(), selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 1);

  table.active = 1;
  EXPECT_EQ(select_profile(table, ProfileRequest::kPinned, 0, decode_input(), selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 0);
}

TEST(ExecuTorchOptimizationProfileSelection, AutoPicksTheOnlyProfileThatFits) {
  ProfileTable table = decode_and_prefill();
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kAuto, 0, prefill_input(), selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 1);
}

// Auto is sticky where the profiles overlap: a one-token input still fits the
// prefill profile, so a decode step after a prefill step stays on profile 1
// rather than dropping back to the lowest matching index. Documented behavior,
// and the reason prefill/decode workloads should pin instead.
TEST(ExecuTorchOptimizationProfileSelection, AutoKeepsTheActiveProfileWhereRangesOverlap) {
  ProfileTable table = decode_and_prefill();
  table.active = 1;
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kAuto, 0, decode_input(), selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 1);
}

TEST(ExecuTorchOptimizationProfileSelection, AutoRescansOnceTheActiveProfileStopsFitting) {
  ProfileTable table = decode_and_prefill();
  table.active = 0;
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kAuto, 0, prefill_input(), selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 1);
}

// A shape no profile covers is an input error, not a silent clamp.
TEST(ExecuTorchOptimizationProfileSelection, AutoRejectsShapeNoProfileCovers) {
  ProfileTable table = decode_and_prefill();
  const std::vector<nvinfer1::Dims> too_long{dims({1, 4096})};
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(table, ProfileRequest::kAuto, 0, too_long, selected), ProfileSelection::kNoProfileMatchesInputs);
}

TEST(ExecuTorchOptimizationProfileSelection, AutoRejectsRankThatDoesNotMatchTheProfile) {
  ProfileTable table = decode_and_prefill();
  const std::vector<nvinfer1::Dims> wrong_rank{dims({1, 1, 1})};
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(table, ProfileRequest::kAuto, 0, wrong_rank, selected), ProfileSelection::kNoProfileMatchesInputs);
}

TEST(ExecuTorchOptimizationProfileSelection, PinningPastTheEndOfAMultiProfileEngineIsRejected) {
  ProfileTable table = decode_and_prefill();
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(table, ProfileRequest::kPinned, 2, decode_input(), selected),
      ProfileSelection::kRequestedProfileUnavailable);
  EXPECT_EQ(
      select_profile(table, ProfileRequest::kPinned, -3, decode_input(), selected),
      ProfileSelection::kRequestedProfileUnavailable);
}

// A .pte may mix a multi-profile engine with a static one. Pinning a nonzero
// profile for the former must not fail the latter, which has one shape and so
// nothing to switch.
TEST(ExecuTorchOptimizationProfileSelection, StaticEngineToleratesAPinItCannotHonor) {
  ProfileTable static_engine;
  static_engine.bounds = {{bounds({1, 16}, {1, 16})}};
  static_engine.all_inputs_static = true;
  const std::vector<nvinfer1::Dims> fixed_input{dims({1, 16})};
  int32_t selected = -1;

  EXPECT_EQ(select_profile(static_engine, ProfileRequest::kPinned, 1, fixed_input, selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 0);
}

// A single-profile *dynamic* engine genuinely cannot honor a nonzero pin, so it
// reports instead of quietly running the wrong regime.
TEST(ExecuTorchOptimizationProfileSelection, SingleProfileDynamicEngineRejectsANonzeroPin) {
  ProfileTable dynamic_engine;
  dynamic_engine.bounds = {{bounds({1, 1}, {1, 2048})}};
  dynamic_engine.all_inputs_static = false;
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(dynamic_engine, ProfileRequest::kPinned, 1, decode_input(), selected),
      ProfileSelection::kRequestedProfileUnavailable);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
