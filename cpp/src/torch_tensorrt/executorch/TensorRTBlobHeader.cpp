#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"

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

// The two scalar fields are found by searching the metadata text for the key,
// quotes included, so anything else quoted the same way would be matched
// instead: a binding named device_id sitting in the alias array the search runs
// over, or a string value that is exactly the key name. Requiring the match to
// be in key position -- its own colon next, whitespace aside -- is what tells
// the two apart, since a value is followed by a comma or a closing brace. A
// value that merely *contains* the key text needs no rule: json.dumps escapes
// the quotes it carries, so the closing quote of the search never lines up and
// there is no match to reject. An occurrence that is not in key position is
// passed over rather than refused: the key may still be ahead of it, and if it
// is not, the field is absent and keeps its default, which is what a blob
// written before the field existed wants.
std::size_t value_pos_after_key(const std::string& json, std::size_t search_from, const char* key) {
  const std::size_t key_len = std::strlen(key);
  std::size_t pos = search_from;
  while ((pos = json.find(key, pos)) != std::string::npos) {
    const std::size_t colon = skip_ws(json, pos + key_len);
    if (colon < json.size() && json[colon] == ':') {
      return skip_ws(json, colon + 1);
    }
    pos += key_len;
  }
  return std::string::npos;
}

bool parse_bool_after_key(const std::string& json, std::size_t search_from, const char* key, bool& value) {
  const std::size_t val = value_pos_after_key(json, search_from, key);
  if (val == std::string::npos) {
    return true;
  }
  if (json.compare(val, 4, "true") == 0) {
    value = true;
    return true;
  }
  if (json.compare(val, 5, "false") == 0) {
    value = false;
    return true;
  }
  return false;
}

bool parse_int_after_key(const std::string& json, std::size_t search_from, const char* key, int& value) {
  std::size_t pos = value_pos_after_key(json, search_from, key);
  if (pos == std::string::npos) {
    return true;
  }
  bool neg = false;
  if (pos < json.size() && json[pos] == '-') {
    neg = true;
    ++pos;
  }
  // Accumulated in 64 bits and bounded on every digit, because the digit count
  // is the blob's to choose. Overflowing an int here is undefined behaviour --
  // a trapping build aborts on it -- and in an ordinary one it wraps, which is
  // the bad case: the only field parsed this way is device_id, and a value just
  // over four billion wraps onto a device that exists, so cudaSetDevice then
  // succeeds and the engine deserializes on a GPU nobody asked for. Anything
  // outside an int is refused instead.
  int64_t parsed = 0;
  bool saw_digit = false;
  while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
    saw_digit = true;
    parsed = parsed * 10 + (json[pos] - '0');
    if (parsed > std::numeric_limits<int>::max()) {
      return false;
    }
    ++pos;
  }
  if (!saw_digit) {
    return false;
  }
  value = static_cast<int>(neg ? -parsed : parsed);
  return true;
}

bool parse_metadata_json(const std::string& json, bool expects_aliased_io, TensorRTBlobHeader& out) {
  out.input_binding_names.clear();
  out.output_binding_names.clear();
  out.aliased_io.clear();
  out.hardware_compatible = false;
  out.device_id = 0;

  // Found the same way as the two scalars below, so all four keys agree about
  // what a key is: an occurrence followed by its own colon, and the array right
  // after that colon rather than the next '[' anywhere in the text. A string
  // value that spells io_bindings is then passed over instead of being taken
  // for the key, and a blob whose io_bindings is not an array is refused rather
  // than walked from some unrelated bracket further on.
  const std::size_t arr_start = value_pos_after_key(json, 0, "\"io_bindings\"");
  if (arr_start == std::string::npos || arr_start >= json.size() || json[arr_start] != '[') {
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
  //
  // Search from pos (past the io_bindings array) so a model input literally
  // named "aliased_io" isn't matched as the array key.
  std::size_t scalars_from = pos;
  std::size_t apos = value_pos_after_key(json, pos, "\"aliased_io\"");
  if (apos != std::string::npos) {
    if (apos >= json.size() || json[apos] != '[') {
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
    // Past the alias array, not merely past io_bindings: the array is written
    // between the two and this walk advances a position of its own, so the
    // scalar scans below would otherwise search across the alias entries and
    // match an aliased binding named like one of their keys.
    scalars_from = apos;
  }

  // The magic says which of the two shapes this blob is, and until here nothing
  // compared that against what was parsed. TR02 means the metadata carries
  // aliased_io, so an empty list is a blob whose alias array this walk did not
  // find -- absent, truncated away, one byte wrong in the key, or written
  // before io_bindings, since the search starts past that array. The threaded
  // shape does not fail afterwards: every aliased output gets storage of its
  // own and the caller's cache quietly stops being updated, which is the
  // outcome the aliasing code exists to make impossible. The converse is not
  // refused: a TR01 blob carrying an alias array is read as aliased and run
  // correctly here, and refusing it would only turn a blob this runtime handles
  // into one it does not.
  if (expects_aliased_io && out.aliased_io.empty()) {
    return false;
  }

  return parse_bool_after_key(json, scalars_from, "\"hardware_compatible\"", out.hardware_compatible) &&
      parse_int_after_key(json, scalars_from, "\"device_id\"", out.device_id);
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
  // Every extent below is checked by subtracting the offset from the bound
  // rather than by adding the size to the offset. engine_size is a 64-bit field
  // read straight from the file, so the sum form wraps: a blob claiming a size
  // just under 2^64 produces a small total, passes, and hands
  // deserializeCudaEngine a pointer plus a length far past the end of the file.
  // The two metadata extents cannot wrap on a 64-bit size_t -- both operands are
  // 32-bit fields, so their sum is at most 2^33 -- but they are written the same
  // way so that the form, not the width of each field, is what makes them safe.
  // The metadata-against-size check below cannot be the sole reason a blob is
  // refused: the other two imply it, because the metadata extent is held inside
  // engine_offset and engine_offset inside size. It is kept anyway, because it
  // is the metadata extent this function goes on to dereference, and bounding
  // that against the file size where it is read does not depend on a chain
  // through a check about the engine.
  if (out.metadata_offset > size || out.metadata_size > size - out.metadata_offset) {
    return false;
  }
  if (out.engine_offset > size || out.engine_size > size - out.engine_offset) {
    return false;
  }
  if (out.metadata_offset > out.engine_offset || out.metadata_size > out.engine_offset - out.metadata_offset) {
    return false;
  }

  std::string json(reinterpret_cast<const char*>(bytes + out.metadata_offset), out.metadata_size);
  return parse_metadata_json(json, aliased_io_magic, out);
}

} // namespace executorch_backend
} // namespace torch_tensorrt
