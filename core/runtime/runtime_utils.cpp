/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Serialization utilities shared by TRTEngine and register_jit_hooks.
 * Kept in a separate translation unit so that the ExecuTorch backend
 * can link these without pulling in the
 * TorchScript torch::class_ / TORCH_LIBRARY registrations in
 * register_jit_hooks.cpp, which would cause a duplicate-registration
 * crash when libtorchtrt.so is also loaded in the same process.
 */

#include <string>
#include <vector>

#include "core/runtime/runtime.h"
#include "core/util/macros.h"
#include "torch_tensorrt/serialization/TensorRTBindingNames.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

std::string serialize_bindings(const std::vector<std::string>& bindings) {
  std::string serialized_binding_info = serialization::serialize_binding_names(bindings);

  LOG_DEBUG("Serialized Binding Info: " << serialized_binding_info);

  return serialized_binding_info;
}

// Base64 alphabet (RFC 4648 §4)
static const std::string sym_table = // NOLINT(cert-err58-cpp)
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"; //=

std::string base64_encode(const std::string& in) {
  std::string out;
  int64_t val = 0, valb = -6;
  for (unsigned char c : in) {
    val = (val << 8) + c;
    valb += 8;
    while (valb >= 0) {
      out.push_back(sym_table[(val >> valb) & 0x3F]);
      valb -= 6;
    }
  }
  if (valb > -6) {
    out.push_back(sym_table[((val << 8) >> (valb + 8)) & 0x3F]);
  }
  while (out.size() % 4) {
    out.push_back('=');
  }
  return out;
}

std::string base64_decode(const std::string& in) {
  std::string out;
  std::vector<int> T(256, -1);
  for (int i = 0; i < 64; i++) {
    T[sym_table[i]] = i;
  }

  int64_t val = 0, valb = -8;
  for (unsigned char c : in) {
    if (T[c] == -1) {
      break;
    }
    val = (val << 6) + T[c];
    valb += 6;
    if (valb >= 0) {
      out.push_back(char((val >> valb) & 0xFF));
      valb -= 8;
    }
  }
  return out;
}

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
