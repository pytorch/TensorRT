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

#include <sstream>
#include <string>
#include <vector>

#include "core/runtime/runtime.h"
#include "core/util/macros.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

std::string serialize_bindings(const std::vector<std::string>& bindings) {
  std::stringstream ss;
  for (size_t i = 0; i < bindings.size() - 1; i++) {
    ss << bindings[i] << TRTEngine::BINDING_DELIM;
  }
  ss << bindings[bindings.size() - 1];

  std::string serialized_binding_info = ss.str();

  LOG_DEBUG("Serialized Binding Info: " << serialized_binding_info);

  return serialized_binding_info;
}

// Aliased I/O wire format:
//   record:  "<output_name>@<input_name>@<kind>"
//   joined:  records separated by TRTEngine::BINDING_DELIM ('%')
// '@' is the intra-record field separator — TRT binding names are
// alphanumeric + underscore so '@' cannot collide with a binding name.
static const char ALIASED_IO_FIELD_DELIM = '@';

std::string serialize_aliased_io(const std::unordered_map<std::string, AliasedIOSpec>& aliased_io) {
  if (aliased_io.empty()) {
    return "";
  }
  std::stringstream ss;
  bool first = true;
  for (const auto& kv : aliased_io) {
    if (!first) {
      ss << TRTEngine::BINDING_DELIM;
    }
    first = false;
    ss << kv.first << ALIASED_IO_FIELD_DELIM << kv.second.input_binding_name << ALIASED_IO_FIELD_DELIM
       << alias_kind_to_string(kv.second.kind);
  }
  std::string out = ss.str();
  LOG_DEBUG("Serialized aliased_io: " << out);
  return out;
}

std::unordered_map<std::string, AliasedIOSpec> deserialize_aliased_io(const std::string& s) {
  std::unordered_map<std::string, AliasedIOSpec> out;
  if (s.empty()) {
    return out;
  }
  size_t pos = 0;
  while (pos < s.size()) {
    size_t rec_end = s.find(TRTEngine::BINDING_DELIM, pos);
    std::string rec = (rec_end == std::string::npos) ? s.substr(pos) : s.substr(pos, rec_end - pos);

    size_t f1 = rec.find(ALIASED_IO_FIELD_DELIM);
    if (f1 == std::string::npos) {
      LOG_WARNING("Skipping malformed aliased_io record (missing first field delim): " << rec);
    } else {
      size_t f2 = rec.find(ALIASED_IO_FIELD_DELIM, f1 + 1);
      if (f2 == std::string::npos) {
        LOG_WARNING("Skipping malformed aliased_io record (missing second field delim): " << rec);
      } else {
        std::string out_name = rec.substr(0, f1);
        std::string in_name = rec.substr(f1 + 1, f2 - f1 - 1);
        std::string kind_str = rec.substr(f2 + 1);
        out[out_name] = AliasedIOSpec{in_name, alias_kind_from_string(kind_str)};
      }
    }
    if (rec_end == std::string::npos)
      break;
    pos = rec_end + 1;
  }
  return out;
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
