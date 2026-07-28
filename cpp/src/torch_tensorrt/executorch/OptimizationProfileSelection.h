/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Which TensorRT optimization profile an execution runs under.
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

// The [min, max] dim envelope one optimization profile allows for one input.
struct InputProfileBounds {
  nvinfer1::Dims min{};
  nvinfer1::Dims max{};
};

// Everything a profile decision depends on, read from the engine once at init().
struct ProfileTable {
  // Indexed [profile][input]. The outer size is the engine's optimization
  // profile count, which is at least 1; a single-profile engine keeps exactly
  // one row and never switches.
  std::vector<std::vector<InputProfileBounds>> bounds;
  // True when every input dim is pinned to one extent in every profile. Such an
  // engine accepts one shape only, so a request for a profile it does not have
  // changes nothing it can do.
  bool all_inputs_static = true;
  // The profile currently loaded into the execution context.
  int32_t active = 0;

  int32_t size() const {
    return static_cast<int32_t>(bounds.size());
  }
};

// What the calling thread asked for, as resolved from OptimizationProfileGuard.
enum class ProfileRequest {
  kUnset, // no guard in scope
  kPinned, // an exact index
  kAuto, // choose from the input shapes
};

enum class ProfileSelection {
  // Created this enum to decouple the profile header from executorch so that we can test it seperately
  kOk,
  // A pinned index this engine does not have, and it is not static enough for
  // that to be harmless.
  kRequestedProfileUnavailable,
  // Auto-selection ran out of profiles.
  kNoProfileMatchesInputs,
};

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

inline bool profile_fits(const ProfileTable& table, int32_t profile, const std::vector<nvinfer1::Dims>& input_dims) {
  const auto& bounds = table.bounds[static_cast<size_t>(profile)];
  for (size_t i = 0; i < input_dims.size(); ++i) {
    if (!dims_fit(input_dims[i], bounds[i])) {
      return false;
    }
  }
  return true;
}

// Resolves one thread's profile request against one engine. `index` is read
// only for ProfileRequest::kPinned.
inline ProfileSelection select_profile(
    const ProfileTable& table,
    ProfileRequest request,
    int32_t index,
    const std::vector<nvinfer1::Dims>& input_dims,
    int32_t& selected) {
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

  // An engine that accepts exactly one shape has nothing to switch, so a request
  // aimed at its multi-profile siblings in the same method is satisfied by
  // profile 0 rather than failing the whole execution. A dynamic engine that
  // lacks the index is a real mismatch and is reported.
  if (index > 0 && table.size() == 1 && table.all_inputs_static) {
    selected = 0;
    return ProfileSelection::kOk;
  }

  return ProfileSelection::kRequestedProfileUnavailable;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
