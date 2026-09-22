#include "torch_tensorrt/executorch/EdgeLLMBlobHeader.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace torch_tensorrt {
namespace executorch_backend {
namespace {

constexpr char EDGE_LLM_MAGIC[4] = {'E', 'L', '0', '1'};
constexpr uint32_t METADATA_OFFSET_FIELD_OFFSET = 4;
constexpr uint32_t METADATA_SIZE_FIELD_OFFSET = 8;
constexpr uint32_t BLOB_OFFSET_FIELD_OFFSET = 12;
constexpr uint32_t BLOB_SIZE_FIELD_OFFSET = 16;
constexpr uint32_t HEADER_SIZE = 32;
constexpr uint32_t BLOB_ALIGNMENT = 16;
constexpr int SUPPORTED_ABI_VERSION = 1;

template <typename T>
T read_field(const uint8_t* data, std::size_t offset) {
  T value{};
  std::memcpy(&value, data + offset, sizeof(T));
  return value;
}

std::size_t skip_ws(const std::string& value, std::size_t pos) {
  while (pos < value.size() && (value[pos] == ' ' || value[pos] == '\t' || value[pos] == '\n' || value[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

std::size_t value_after_key(const std::string& json, const char* key) {
  const std::string quoted_key = std::string("\"") + key + "\"";
  const std::size_t key_pos = json.find(quoted_key);
  if (key_pos == std::string::npos) {
    return std::string::npos;
  }
  const std::size_t colon = json.find(':', key_pos + quoted_key.size());
  if (colon == std::string::npos) {
    return std::string::npos;
  }
  return skip_ws(json, colon + 1);
}

bool parse_int(const std::string& json, const char* key, int& out) {
  std::size_t pos = value_after_key(json, key);
  if (pos == std::string::npos || pos >= json.size()) {
    return false;
  }
  bool negative = false;
  if (json[pos] == '-') {
    negative = true;
    ++pos;
  }
  int value = 0;
  bool saw_digit = false;
  while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
    saw_digit = true;
    if (value > (std::numeric_limits<int>::max() - (json[pos] - '0')) / 10) {
      return false;
    }
    value = value * 10 + (json[pos] - '0');
    ++pos;
  }
  if (!saw_digit) {
    return false;
  }
  out = negative ? -value : value;
  return true;
}

bool parse_string(const std::string& json, const char* key, std::string& out) {
  std::size_t pos = value_after_key(json, key);
  if (pos == std::string::npos || pos >= json.size() || json[pos] != '"') {
    return false;
  }
  ++pos;
  out.clear();
  while (pos < json.size()) {
    if (json[pos] == '"') {
      return true;
    }
    if (json[pos] == '\\') {
      ++pos;
      if (pos >= json.size()) {
        return false;
      }
    }
    out.push_back(json[pos++]);
  }
  return false;
}

bool parse_compound(const std::string& json, const char* key, char open, char close, std::string& out) {
  std::size_t pos = value_after_key(json, key);
  if (pos == std::string::npos || pos >= json.size() || json[pos] != open) {
    return false;
  }
  const std::size_t start = pos;
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (; pos < json.size(); ++pos) {
    const char ch = json[pos];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        in_string = false;
      }
      continue;
    }
    if (ch == '"') {
      in_string = true;
    } else if (ch == open) {
      ++depth;
    } else if (ch == close) {
      --depth;
      if (depth == 0) {
        out = json.substr(start, pos - start + 1);
        return true;
      }
    }
  }
  return false;
}

bool parse_metadata(const std::string& json, EdgeLLMBlobHeader& out) {
  std::string outputs_json;
  return parse_int(json, "abi_version", out.abi_version) && out.abi_version == SUPPORTED_ABI_VERSION &&
      parse_string(json, "component", out.component) && out.component == "vision" &&
      parse_string(json, "runner", out.runner) && out.runner == "vit" &&
      parse_compound(json, "outputs", '[', ']', outputs_json) && outputs_json != "[]" &&
      parse_compound(json, "runner_config", '{', '}', out.runner_config_json);
}

} // namespace

const void* EdgeLLMBlobHeader::nested_blob_data(const void* payload, const EdgeLLMBlobHeader& h) {
  return static_cast<const uint8_t*>(payload) + h.blob_offset;
}

bool EdgeLLMBlobHeader::parse(const void* data, std::size_t size, EdgeLLMBlobHeader& out) {
  if (data == nullptr || size < HEADER_SIZE) {
    return false;
  }
  const auto* bytes = static_cast<const uint8_t*>(data);
  if (std::memcmp(bytes, EDGE_LLM_MAGIC, sizeof(EDGE_LLM_MAGIC)) != 0) {
    return false;
  }

  out = EdgeLLMBlobHeader{};
  out.metadata_offset = read_field<uint32_t>(bytes, METADATA_OFFSET_FIELD_OFFSET);
  out.metadata_size = read_field<uint32_t>(bytes, METADATA_SIZE_FIELD_OFFSET);
  out.blob_offset = read_field<uint32_t>(bytes, BLOB_OFFSET_FIELD_OFFSET);
  out.blob_size = read_field<uint64_t>(bytes, BLOB_SIZE_FIELD_OFFSET);

  const uint64_t metadata_end = static_cast<uint64_t>(out.metadata_offset) + out.metadata_size;
  const uint64_t blob_end = static_cast<uint64_t>(out.blob_offset) + out.blob_size;
  if (out.metadata_offset < HEADER_SIZE || out.blob_offset % BLOB_ALIGNMENT != 0 || metadata_end > out.blob_offset ||
      blob_end > size) {
    return false;
  }

  out.metadata_json.assign(
      reinterpret_cast<const char*>(bytes + out.metadata_offset), static_cast<std::size_t>(out.metadata_size));
  return parse_metadata(out.metadata_json, out);
}

} // namespace executorch_backend
} // namespace torch_tensorrt
