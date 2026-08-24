/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * @file OptimizationProfileSelection.h
 * @brief Which TensorRT optimization profile an execution runs under.
 *
 * Kept free of ExecuTorch, CUDA, and the engine itself so the policy can be
 * exercised without a GPU; reporting the outcome is left to the caller.
 */
#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

/// @brief The [min, max] dim envelope one optimization profile allows for one input.
struct InputProfileBounds {
  nvinfer1::Dims min{}; ///< Smallest shape the profile accepts.
  nvinfer1::Dims max{}; ///< Largest shape the profile accepts.
};

/// @brief Everything a profile decision depends on, read from the engine once at init().
struct ProfileTable {
  /**
   * @brief Bounds indexed [profile][input].
   *
   * The outer size is the engine's optimization profile count, which is at
   * least 1; a single-profile engine keeps exactly one row and never switches.
   */
  std::vector<std::vector<InputProfileBounds>> bounds;
  /// @brief The profile currently loaded into the execution context.
  int32_t active = 0;

  /// @return The engine's optimization profile count.
  int32_t size() const {
    return static_cast<int32_t>(bounds.size());
  }
};

/// @brief What the calling thread asked for, as resolved from OptimizationProfileGuard.
enum class ProfileRequest {
  kUnset, ///< No guard in scope.
  kPinned, ///< An exact index.
  kAuto, ///< Choose from the input shapes.
};

/**
 * @brief The outcome of one profile decision.
 *
 * Its own enum rather than executorch's Error so that this header stays
 * independent of executorch and can be tested separately.
 *
 * Two axes: whether execution continues, and which message the caller prints.
 * The two failure values stay apart rather than being merged and re-derived from
 * the request kind, because the empty-table guard in select_profile() returns
 * kNoProfileMatchesInputs for every request kind -- so one merged value would put
 * the message back at the mercy of which branches each request can reach.
 */
enum class ProfileSelection {
  kOk, ///< The selected profile is usable.
  /**
   * @brief Succeeded, but the pin could not be honored and profile 0 was used instead.
   *
   * Distinct from kOk so the caller can warn that the pin did nothing here.
   */
  kPinIgnoredSingleProfile,
  /**
   * @brief A pinned index this engine does not have and cannot substitute for.
   *
   * Fatal, on the same split the standard runtime uses in
   * TRTEngine::set_active_profile_with_stream: an engine that had profiles to
   * choose between must not quietly run on one the caller did not ask for.
   * OptimizationProfileGuard cannot validate anything -- it never sees an
   * engine, by design -- so execute() is the only place an ExecuTorch caller's
   * bad index can be caught at all.
   */
  kRequestedProfileUnavailable,
  /// @brief Auto-selection ran out of profiles.
  kNoProfileMatchesInputs,
};

/**
 * @brief Whether one input shape falls inside one profile's envelope.
 *
 * @param dims The runtime shape of the input.
 * @param bounds The profile's [min, max] envelope for that input.
 * @return true when the ranks match and every extent is in range.
 */
inline bool dims_fit(const nvinfer1::Dims& dims, const InputProfileBounds& bounds) {
  if (dims.nbDims != bounds.min.nbDims) {
    return false;
  }
  for (int d = 0; d < dims.nbDims; ++d) {
    if (dims.d[d] < bounds.min.d[d] || dims.d[d] > bounds.max.d[d]) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Whether one profile accepts every input shape of an execution.
 *
 * A profile index or bounds row that cannot describe these inputs answers "does
 * not fit" rather than indexing past the end of the table. Neither is reachable
 * from the backend, which builds one row of num_inputs bounds per profile and
 * only ever stores an index this function approved -- but this is installed
 * public API, and on the auto path the answer also happens to be the useful one:
 * a ProfileTable carrying a stale ProfileTable::active rescans from 0 instead of
 * crashing.
 *
 * @param table The engine's profile bounds.
 * @param profile Index of the profile to test.
 * @param input_dims Runtime shape of each input, in binding order.
 * @return true when the profile accepts all of them.
 */
inline bool profile_fits(const ProfileTable& table, int32_t profile, const std::vector<nvinfer1::Dims>& input_dims) {
  if (profile < 0 || profile >= table.size()) {
    return false;
  }
  const auto& bounds = table.bounds[static_cast<size_t>(profile)];
  if (bounds.size() < input_dims.size()) {
    return false;
  }
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (!dims_fit(input_dims[i], bounds[i])) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Resolves one thread's profile request against one engine.
 *
 * @param table The engine's profile bounds and currently loaded profile.
 * @param request What the calling thread asked for.
 * @param index The pinned profile index; read only for ProfileRequest::kPinned.
 * @param input_dims Runtime shape of each input, in binding order.
 * @param[out] selected The profile to run, written only when the result allows
 *   execution to continue (ProfileSelection::kOk or
 *   ProfileSelection::kPinIgnoredSingleProfile).
 * @return What the caller should do, and which message it should print.
 */
inline ProfileSelection select_profile(
    const ProfileTable& table,
    ProfileRequest request,
    int32_t index,
    const std::vector<nvinfer1::Dims>& input_dims,
    int32_t& selected) {
  // init() rejects an engine reporting no profiles, so this is unreachable in the
  // backend. Checked here so the policy is safe to call on its own rather than on
  // the strength of a guard in another translation unit.
  if (table.bounds.empty()) {
    return ProfileSelection::kNoProfileMatchesInputs;
  }

  if (request == ProfileRequest::kUnset) {
    selected = 0;
    return ProfileSelection::kOk;
  }

  if (request == ProfileRequest::kAuto) {
    // Sticky first-fit: keep the loaded profile while it still fits, so shapes
    // that alternate between two equally valid profiles don't thrash the
    // context. Only rescan from 0 once it stops fitting. Overlapping profiles
    // therefore resolve by history, not by lowest index; pin explicitly when
    // that matters.
    if (profile_fits(table, table.active, input_dims)) {
      selected = table.active;
      return ProfileSelection::kOk;
    }
    for (int32_t p = 0; p < table.size(); ++p) {
      if (profile_fits(table, p, input_dims)) {
        selected = p;
        return ProfileSelection::kOk;
      }
    }
    return ProfileSelection::kNoProfileMatchesInputs;
  }

  if (index >= 0 && index < table.size()) {
    selected = index;
    return ProfileSelection::kOk;
  }

  // A single-profile engine has no choice to get wrong: profile 0 is the only
  // thing it can run, whether or not its shapes are dynamic. So a pin aimed at a
  // multi-profile sibling in the same method must not fail it. An engine with
  // several profiles is different -- substituting one would be a guess -- so an
  // index it lacks stays an error there.
  if (index > 0 && table.size() == 1) {
    selected = 0;
    return ProfileSelection::kPinIgnoredSingleProfile;
  }

  return ProfileSelection::kRequestedProfileUnavailable;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
