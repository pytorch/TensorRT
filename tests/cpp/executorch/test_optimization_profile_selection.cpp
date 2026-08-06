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
 *
 * Deliberately out of scope, because all of it needs a live engine: applying the
 * decision (setOptimizationProfileAsync), writing profiles.active back, the
 * ordering that puts the switch before setInputShape, and mark_inflight. Those
 * belong to the end-to-end ExecuTorch tests.
 */

#include "torch_tensorrt/executorch/OptimizationProfileSelection.h"

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

// A profile has to fit *every* input, not just the first one it is asked about.
// Here profile 0 accepts the one-token input_ids but not the longer second input,
// so auto has to keep looking rather than stop at the first partial match.
TEST(ExecuTorchOptimizationProfileSelection, AutoSkipsAProfileThatFitsOnlySomeInputs) {
  ProfileTable table;
  table.bounds = {
      {bounds({1, 1}, {1, 1}), bounds({1, 1}, {1, 1})}, // profile 0: second input too narrow
      {bounds({1, 1}, {1, 1}), bounds({1, 1}, {1, 128})}, // profile 1: fits both
  };
  const std::vector<nvinfer1::Dims> inputs{dims({1, 1}), dims({1, 64})};
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kAuto, 0, inputs, selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 1);
}

// The rescan is a first-fit from 0, so when the active profile stops fitting and
// several others would serve, the lowest matching index wins.
TEST(ExecuTorchOptimizationProfileSelection, RescanTakesTheLowestOfSeveralMatchingProfiles) {
  ProfileTable table;
  table.bounds = {
      {bounds({1, 1}, {1, 64})}, // profile 0: fits
      {bounds({1, 1}, {1, 256})}, // profile 1: fits too
      {bounds({1, 512}, {1, 2048})}, // profile 2: active, no longer fits
  };
  table.active = 2;
  const std::vector<nvinfer1::Dims> short_input{dims({1, 32})};
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kAuto, 0, short_input, selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 0);
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

// A .pte may mix a multi-profile engine with a single-profile one. Pinning a
// nonzero profile for the former must not fail the latter, which has profile 0
// and nothing to switch to. Reported as kPinIgnoredSingleProfile rather than
// kOk so execute() can say the pin did nothing here.
TEST(ExecuTorchOptimizationProfileSelection, SingleProfileEngineToleratesAPinItCannotHonor) {
  ProfileTable static_engine;
  static_engine.bounds = {{bounds({1, 16}, {1, 16})}};
  const std::vector<nvinfer1::Dims> fixed_input{dims({1, 16})};
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(static_engine, ProfileRequest::kPinned, 1, fixed_input, selected),
      ProfileSelection::kPinIgnoredSingleProfile);
  EXPECT_EQ(selected, 0);
}

// Whether the single profile's shapes are dynamic makes no difference: profile 0
// is still the only thing the engine can run, so it is tolerated the same way.
TEST(ExecuTorchOptimizationProfileSelection, SingleProfileToleranceDoesNotDependOnDynamicShapes) {
  ProfileTable dynamic_engine;
  dynamic_engine.bounds = {{bounds({1, 1}, {1, 2048})}};
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(dynamic_engine, ProfileRequest::kPinned, 1, decode_input(), selected),
      ProfileSelection::kPinIgnoredSingleProfile);
  EXPECT_EQ(selected, 0);
}

// The tolerance covers indices the engine merely lacks, not nonsense ones. A
// negative index is the shape a failed lookup returns, so it has to be reported
// even here, where profile 0 would otherwise be a tempting substitute. This is
// what the removed kAutoSelectProfile == -1 sentinel used to swallow.
TEST(ExecuTorchOptimizationProfileSelection, SingleProfileEngineStillRejectsANegativeIndex) {
  ProfileTable table;
  table.bounds = {{bounds({1, 16}, {1, 16})}};
  const std::vector<nvinfer1::Dims> fixed_input{dims({1, 16})};
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(table, ProfileRequest::kPinned, -1, fixed_input, selected),
      ProfileSelection::kRequestedProfileUnavailable);
}

// The tolerance stops at one profile. With several, substituting profile 0 would
// be a guess about which regime the caller wanted, so this stays an error.
TEST(ExecuTorchOptimizationProfileSelection, MultiProfileEngineDoesNotSubstituteForAMissingIndex) {
  ProfileTable table;
  table.bounds = {
      {bounds({1, 1}, {1, 1})},
      {bounds({1, 1}, {1, 128})},
  };
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(table, ProfileRequest::kPinned, 2, decode_input(), selected),
      ProfileSelection::kRequestedProfileUnavailable);
}

// An engine with no inputs has nothing to constrain the choice, so every profile
// trivially fits and auto keeps the loaded one.
TEST(ExecuTorchOptimizationProfileSelection, AutoHandlesAnEngineWithNoInputs) {
  ProfileTable table;
  table.bounds = {{}, {}};
  table.active = 1;
  int32_t selected = -1;

  EXPECT_EQ(select_profile(table, ProfileRequest::kAuto, 0, {}, selected), ProfileSelection::kOk);
  EXPECT_EQ(selected, 1);
}

// A table with no profiles at all is malformed; init() rejects such an engine
// before execute() ever runs. Guarded here anyway so the policy never indexes an
// empty bounds vector on the strength of a check in another translation unit.
TEST(ExecuTorchOptimizationProfileSelection, EmptyTableIsRejectedRatherThanIndexed) {
  ProfileTable empty;
  int32_t selected = -1;

  EXPECT_EQ(
      select_profile(empty, ProfileRequest::kUnset, 0, decode_input(), selected),
      ProfileSelection::kNoProfileMatchesInputs);
  EXPECT_EQ(
      select_profile(empty, ProfileRequest::kAuto, 0, decode_input(), selected),
      ProfileSelection::kNoProfileMatchesInputs);
  EXPECT_EQ(
      select_profile(empty, ProfileRequest::kPinned, 0, decode_input(), selected),
      ProfileSelection::kNoProfileMatchesInputs);
}

} // namespace
} // namespace executorch_backend
} // namespace torch_tensorrt
