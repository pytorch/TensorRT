/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

// TR02 marks a blob whose metadata carries aliased_io; TR01 is one without.
// This parser handles aliased_io, so it accepts either, and holds TR02 to its
// promise: see the cross-check at the end of parse_metadata_json.
constexpr char TENSORRT_MAGIC[4] = {'T', 'R', '0', '1'};
constexpr char TENSORRT_MAGIC_ALIASED_IO[4] = {'T', 'R', '0', '2'};
constexpr uint32_t METADATA_OFFSET_FIELD_OFFSET = 4;
constexpr uint32_t METADATA_SIZE_FIELD_OFFSET = 8;
constexpr uint32_t ENGINE_OFFSET_FIELD_OFFSET = 12;
constexpr uint32_t ENGINE_SIZE_FIELD_OFFSET = 16;
constexpr uint32_t HEADER_SIZE = 32;
constexpr uint32_t ENGINE_ALIGNMENT = 16;

std::size_t skip_ws(const std::string& s, std::size_t pos) {
  while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

// A backslash is dropped and the character after it kept. That is the correct
// decoding for \" \\ and \/ and the wrong one for every other escape, which a
// JSON reader turns into a control character, a codepoint, or an error. Nothing
// this parser reads can carry one. Every string it compares is either a key or
// an alias kind, which TensorRTBlobMetadata.to_json writes as literals, or a
// binding name, and on the only path that writes these blobs an input binding is
// named for its Edge placeholder -- which torch.export has already reduced to
// [A-Za-z0-9_] -- and an output binding is output0..N. So json.dumps has nothing
// to escape, and the values skip_value walks past need only their end, which
// this finds either way.
std::size_t parse_string(const std::string& s, std::size_t pos, std::string& out) {
  if (pos >= s.size() || s[pos] != '"') {
    return std::string::npos;
  }
  ++pos;
  out.clear();
  while (pos < s.size() && s[pos] != '"') {
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      ++pos;
    }
    out += s[pos++];
  }
  if (pos >= s.size()) {
    return std::string::npos;
  }
  return pos + 1;
}

std::size_t skip_value(const std::string& s, std::size_t pos) {
  pos = skip_ws(s, pos);
  if (pos >= s.size()) {
    return std::string::npos;
  }

  if (s[pos] == '"') {
    std::string unused;
    return parse_string(s, pos, unused);
  }
  if (s[pos] == '{' || s[pos] == '[') {
    const char open = s[pos];
    const char close = open == '{' ? '}' : ']';
    int depth = 1;
    ++pos;
    while (pos < s.size() && depth > 0) {
      if (s[pos] == '"') {
        std::string unused;
        pos = parse_string(s, pos, unused);
        if (pos == std::string::npos) {
          return pos;
        }
        continue;
      }
      if (s[pos] == open) {
        ++depth;
      } else if (s[pos] == close) {
        --depth;
      }
      ++pos;
    }
    return pos;
  }

  while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']' && s[pos] != ' ' && s[pos] != '\t' &&
         s[pos] != '\n' && s[pos] != '\r') {
    ++pos;
  }
  return pos;
}

// A reader that stops short of the delimiter silently accepts a different value, so 3.5 reads as 3.
bool ends_value(const std::string& json, std::size_t pos) {
  if (pos >= json.size()) {
    return true;
  }
  const char c = json[pos];
  return c == ',' || c == '}' || c == ']' || std::isspace(static_cast<unsigned char>(c));
}

bool parse_bool_after_key(const std::string& json, std::size_t search_from, const char* key, bool& value) {
  const std::size_t key_pos = json.find(key, search_from);
  if (key_pos == std::string::npos) {
    return true;
  }
  const std::size_t colon = json.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  const std::size_t val = skip_ws(json, colon + 1);
  if (json.compare(val, 4, "true") == 0 && ends_value(json, val + 4)) {
    value = true;
    return true;
  }
  if (json.compare(val, 5, "false") == 0 && ends_value(json, val + 5)) {
    value = false;
    return true;
  }
  return false;
}

bool parse_int_after_key(const std::string& json, std::size_t search_from, const char* key, int& value) {
  const std::size_t key_pos = json.find(key, search_from);
  if (key_pos == std::string::npos) {
    return true;
  }
  const std::size_t colon = json.find(':', key_pos);
  if (colon == std::string::npos) {
    return false;
  }
  std::size_t pos = skip_ws(json, colon + 1);
  bool neg = false;
  if (pos < json.size() && json[pos] == '-') {
    neg = true;
    ++pos;
  }
  // Digits come from the file, so accumulate wide and refuse what will not fit.
  constexpr int64_t MAX_MAGNITUDE = -static_cast<int64_t>(std::numeric_limits<int>::min());
  int64_t parsed = 0;
  bool saw_digit = false;
  while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
    saw_digit = true;
    parsed = parsed * 10 + (json[pos] - '0');
    if (parsed > MAX_MAGNITUDE) {
      return false;
    }
    ++pos;
  }
  if (!saw_digit) {
    return false;
  }
  if (!ends_value(json, pos)) {
    return false;
  }
  const int64_t signed_value = neg ? -parsed : parsed;
  if (signed_value < std::numeric_limits<int>::min() || signed_value > std::numeric_limits<int>::max()) {
    return false;
  }
  value = static_cast<int>(signed_value);
  return true;
}

// A key of the outermost object, so a tensor named after a key is not mistaken for it.
std::size_t find_top_level_key(const std::string& json, const char* key) {
  const std::size_t key_len = std::strlen(key);
  int depth = 0;
  bool in_string = false;
  for (std::size_t at = 0; at < json.size(); ++at) {
    const char c = json[at];
    if (in_string) {
      if (c == '\\') {
        ++at;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      // A name, not a value, and only when it belongs to the outermost object. The colon after it is
      // what distinguishes the two, since a value is a string in exactly the same shape.
      if (depth == 1 && json.compare(at, key_len, key) == 0) {
        std::size_t after = at + key_len;
        while (after < json.size() && std::isspace(static_cast<unsigned char>(json[after]))) {
          ++after;
        }
        if (after < json.size() && json[after] == ':') {
          return at;
        }
      }
      in_string = true;
      continue;
    }
    if (c == '{' || c == '[') {
      ++depth;
    } else if (c == '}' || c == ']') {
      --depth;
    }
  }
  return std::string::npos;
}

bool parse_metadata_json(const std::string& json, bool expects_aliased_io, TensorRTBlobHeader& out) {
  out.input_binding_names.clear();
  out.output_binding_names.clear();
  out.aliased_io.clear();
  out.hardware_compatible = false;
  out.device_id = 0;

  const std::size_t bindings_pos = find_top_level_key(json, "\"io_bindings\"");
  if (bindings_pos == std::string::npos) {
    return false;
  }
  // The array has to be the value of the key, not merely the next '[' in the text.
  // A blob whose io_bindings is an object would otherwise be walked from an
  // unrelated bracket further on, and whatever entries sit there get recorded as
  // this engine's bindings, which is a wrong answer rather than a refusal.
  const std::size_t bindings_colon = json.find(':', bindings_pos);
  if (bindings_colon == std::string::npos) {
    return false;
  }
  const std::size_t arr_start = skip_ws(json, bindings_colon + 1);
  if (arr_start >= json.size() || json[arr_start] != '[') {
    return false;
  }

  // A TensorRT engine has one name space for its tensors, so a name repeated
  // across io_bindings -- in either list -- cannot be two bindings. It is
  // accepted by every name lookup, which stops at the first match, and then
  // contradicted by every address bind, which is keyed on the name and so
  // overwrites whatever the earlier slot bound. For an aliased output the
  // address overwritten is the caller's buffer, and the engine's in-place
  // update lands somewhere else. Refuse the blob here, where the repeat is
  // visible from the bytes alone.
  std::unordered_set<std::string> claimed_bindings;
  std::size_t pos = arr_start + 1;
  while (true) {
    pos = skip_ws(json, pos);
    if (pos >= json.size()) {
      return false;
    }
    if (json[pos] == ']') {
      ++pos;
      break;
    }
    if (json[pos] == ',') {
      ++pos;
      continue;
    }
    if (json[pos] != '{') {
      return false;
    }
    ++pos;

    std::string name;
    bool is_input = false;
    bool saw_is_input = false;

    while (true) {
      pos = skip_ws(json, pos);
      if (pos >= json.size()) {
        return false;
      }
      if (json[pos] == '}') {
        ++pos;
        break;
      }
      if (json[pos] == ',') {
        ++pos;
        continue;
      }

      std::string key;
      pos = parse_string(json, pos, key);
      if (pos == std::string::npos) {
        return false;
      }
      pos = skip_ws(json, pos);
      if (pos >= json.size() || json[pos] != ':') {
        return false;
      }
      pos = skip_ws(json, pos + 1);

      if (key == "name") {
        pos = parse_string(json, pos, name);
        if (pos == std::string::npos) {
          return false;
        }
      } else if (key == "is_input") {
        saw_is_input = true;
        if (json.compare(pos, 4, "true") == 0) {
          is_input = true;
          pos += 4;
        } else if (json.compare(pos, 5, "false") == 0) {
          is_input = false;
          pos += 5;
        } else {
          return false;
        }
      } else {
        pos = skip_value(json, pos);
        if (pos == std::string::npos) {
          return false;
        }
      }
    }

    // A nameless entry cannot be refused earlier because the keys may arrive in
    // any order, so it is refused here, beside the repeat. Skipping it instead
    // would shorten the recorded list while the delegate's argument list keeps
    // its full length, and the two are only inferred from the engine when both
    // are empty -- so one real name beside a blank leaves a short list that no
    // longer lines up with the engine's bindings. An absent "name" key needs no
    // flag of its own, unlike is_input below: the string is still empty here.
    if (name.empty()) {
      return false;
    }
    // is_input has no safe default, so an entry without it is refused beside
    // the nameless one. The initializer here reads the binding as an output
    // and TensorRTIOBinding.is_input in serialization.py defaults to an input,
    // so the two readers of these bytes would disagree -- and one binding
    // changing list shifts every index after it, which on a static-shape
    // engine leaves nothing for shape inference to object to: it runs on the
    // wrong tensors. A key misspelled by one byte is the same case, since it
    // falls through to skip_value and leaves the initializer standing. The
    // writer always emits the key.
    if (!saw_is_input) {
      return false;
    }
    if (!claimed_bindings.insert(name).second) {
      return false;
    }
    if (is_input) {
      out.input_binding_names.push_back(name);
    } else {
      out.output_binding_names.push_back(name);
    }
  }

  // Optional aliased_io array: [{"output":..,"input":..,"kind":..}, ...].
  // Absent in older blobs -> leave empty (backward compatible). Mirrors the
  // io_bindings walk above using the same string helpers.
  const std::size_t alias_key = find_top_level_key(json, "\"aliased_io\"");
  if (alias_key != std::string::npos) {
    std::size_t apos = json.find('[', alias_key);
    if (apos == std::string::npos) {
      return false;
    }
    ++apos;
    std::unordered_set<std::string> claimed_outputs;
    std::unordered_set<std::string> claimed_inputs;
    while (true) {
      apos = skip_ws(json, apos);
      if (apos >= json.size()) {
        return false;
      }
      if (json[apos] == ']') {
        ++apos;
        break;
      }
      if (json[apos] == ',') {
        ++apos;
        continue;
      }
      if (json[apos] != '{') {
        return false;
      }
      ++apos;

      AliasedBinding ab;
      while (true) {
        apos = skip_ws(json, apos);
        if (apos >= json.size()) {
          return false;
        }
        if (json[apos] == '}') {
          ++apos;
          break;
        }
        if (json[apos] == ',') {
          ++apos;
          continue;
        }
        std::string key;
        apos = parse_string(json, apos, key);
        if (apos == std::string::npos) {
          return false;
        }
        apos = skip_ws(json, apos);
        if (apos >= json.size() || json[apos] != ':') {
          return false;
        }
        apos = skip_ws(json, apos + 1);
        if (key == "output") {
          apos = parse_string(json, apos, ab.output);
        } else if (key == "input") {
          apos = parse_string(json, apos, ab.input);
        } else if (key == "kind") {
          apos = parse_string(json, apos, ab.kind);
        } else {
          apos = skip_value(json, apos);
        }
        if (apos == std::string::npos) {
          return false;
        }
      }
      // An entry missing either name is refused rather than skipped, for the
      // reason the binding walk above gives and one more: init() counts the
      // entries it accepts and execute() subtracts that count from the delegate
      // argument list, so a dropped entry surfaces as an argument-count error at
      // every execute, which never mentions aliasing, instead of at parse, which
      // the blob-header tests reach without a GPU.
      if (ab.output.empty() || ab.input.empty()) {
        return false;
      }
      // An output binding may be claimed by at most one entry. A second entry
      // for the same output names the same binding, so nothing that resolves
      // the names can tell the two apart -- but a reader that counts aliased
      // outputs per entry, as TensorRTBackend does to size the delegate
      // argument list, counts one output twice. Refuse the blob here, where
      // the repeat is visible from the bytes alone.
      if (!claimed_outputs.insert(ab.output).second) {
        return false;
      }
      // An input may likewise be claimed by at most one entry. Two entries
      // naming different outputs and one input record the same input index
      // for both, and execute() binds each aliased output to that index's
      // caller pointer -- one address with two writers, so whichever the
      // engine writes second wins and the other update disappears with no
      // error. TensorRT's own aliasing rules out the kv_cache_update kind
      // (init cross-checks it against getAliasedInputTensor), but the user
      // kind is only compared on shape, so two same-shaped outputs onto one
      // input would pass.
      if (!claimed_inputs.insert(ab.input).second) {
        return false;
      }
      // The current Python serializer always writes "kind" (serialization.py),
      // and older blobs carry no aliased_io array at all, so this default is
      // defensive: it only fires for a blob that has an aliased_io entry but
      // omits "kind". Default to the TRT-enforced kind so init()'s kind
      // validation treats an absent key the same as the Python runtime rather
      // than rejecting it as unknown.
      if (ab.kind.empty()) {
        ab.kind = "kv_cache_update";
      }
      out.aliased_io.push_back(std::move(ab));
    }
  }

  // The magic says which of the two shapes this blob is, and until here nothing
  // compared that against what was parsed. TR02 means the metadata carries
  // aliased_io, so an empty list is a blob whose alias array this walk did not
  // find: absent, truncated away, or one byte wrong in the key. The threaded shape
  // does not fail afterwards, it quietly gives every aliased output storage of its
  // own and stops updating the caller's cache, which is the outcome the aliasing
  // code exists to make impossible. The converse is not refused: a TR01 blob
  // carrying an alias array is read as aliased and run correctly here, and
  // refusing it would only turn a blob this runtime handles into one it does not.
  if (expects_aliased_io && out.aliased_io.empty()) {
    return false;
  }
  const std::size_t hw_key = find_top_level_key(json, "\"hardware_compatible\"");
  const std::size_t device_key = find_top_level_key(json, "\"device_id\"");
  return parse_bool_after_key(json, hw_key, "\"hardware_compatible\"", out.hardware_compatible) &&
      parse_int_after_key(json, device_key, "\"device_id\"", out.device_id);
}

} // namespace

const void* TensorRTBlobHeader::engine_data(const void* blob, const TensorRTBlobHeader& h) {
  return static_cast<const uint8_t*>(blob) + h.engine_offset;
}

bool TensorRTBlobHeader::parse(const void* data, std::size_t size, TensorRTBlobHeader& out) {
  if (data == nullptr || size < HEADER_SIZE) {
    return false;
  }

  const auto* bytes = static_cast<const uint8_t*>(data);
  const bool aliased_io_magic = std::memcmp(bytes, TENSORRT_MAGIC_ALIASED_IO, sizeof(TENSORRT_MAGIC_ALIASED_IO)) == 0;
  if (!aliased_io_magic && std::memcmp(bytes, TENSORRT_MAGIC, sizeof(TENSORRT_MAGIC)) != 0) {
    return false;
  }

  auto read_u32 = [&](std::size_t offset) {
    uint32_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
  };
  auto read_u64 = [&](std::size_t offset) {
    uint64_t value = 0;
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
  };

  out.metadata_offset = read_u32(METADATA_OFFSET_FIELD_OFFSET);
  out.metadata_size = read_u32(METADATA_SIZE_FIELD_OFFSET);
  out.engine_offset = read_u32(ENGINE_OFFSET_FIELD_OFFSET);
  out.engine_size = read_u64(ENGINE_SIZE_FIELD_OFFSET);

  if (out.metadata_offset < HEADER_SIZE) {
    return false;
  }
  if (out.engine_offset % ENGINE_ALIGNMENT != 0) {
    return false;
  }
  // Against the space left, not offset plus size, which wraps on a 64 bit size.
  if (out.metadata_offset > size || out.metadata_size > size - out.metadata_offset) {
    return false;
  }
  if (out.engine_offset > size || out.engine_size > size - out.engine_offset) {
    return false;
  }
  if (out.metadata_offset > out.engine_offset ||
      out.metadata_size > static_cast<std::size_t>(out.engine_offset) - out.metadata_offset) {
    return false;
  }

  std::string json(reinterpret_cast<const char*>(bytes + out.metadata_offset), out.metadata_size);
  return parse_metadata_json(json, aliased_io_magic, out);
}

} // namespace executorch_backend
} // namespace torch_tensorrt
