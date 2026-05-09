#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

static constexpr char kTensorRTMagic[4] = {'T', 'R', '0', '1'};
static constexpr uint32_t kHeaderSize = 32;

struct TensorRTBlobHeader {
  uint32_t metadata_offset = 0;
  uint32_t metadata_size = 0;
  uint32_t engine_offset = 0;
  uint64_t engine_size = 0;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  bool hardware_compatible = false;
  int device_id = 0;

  static const void* engine_data(const void* blob, const TensorRTBlobHeader& h) {
    return static_cast<const uint8_t*>(blob) + h.engine_offset;
  }

  static bool parse(const void* data, size_t size, TensorRTBlobHeader& out) {
    if (data == nullptr || size < kHeaderSize) {
      return false;
    }

    const auto* bytes = static_cast<const uint8_t*>(data);
    if (std::memcmp(bytes, kTensorRTMagic, sizeof(kTensorRTMagic)) != 0) {
      return false;
    }

    auto read_u32 = [&](size_t offset) {
      uint32_t value = 0;
      std::memcpy(&value, bytes + offset, sizeof(value));
      return value;
    };
    auto read_u64 = [&](size_t offset) {
      uint64_t value = 0;
      std::memcpy(&value, bytes + offset, sizeof(value));
      return value;
    };

    out.metadata_offset = read_u32(4);
    out.metadata_size = read_u32(8);
    out.engine_offset = read_u32(12);
    out.engine_size = read_u64(16);

    if (out.metadata_offset < kHeaderSize) {
      return false;
    }
    if (out.engine_offset % 16 != 0) {
      return false;
    }
    if (static_cast<size_t>(out.metadata_offset) + out.metadata_size > size) {
      return false;
    }
    if (static_cast<size_t>(out.engine_offset) + out.engine_size > size) {
      return false;
    }
    if (static_cast<size_t>(out.metadata_offset) + out.metadata_size > out.engine_offset) {
      return false;
    }

    std::string json(reinterpret_cast<const char*>(bytes + out.metadata_offset), out.metadata_size);
    return parse_metadata_json(json, out);
  }

 private:
  static size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
      ++pos;
    }
    return pos;
  }

  static size_t parse_string(const std::string& s, size_t pos, std::string& out) {
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

  static size_t skip_value(const std::string& s, size_t pos) {
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

  static bool parse_bool_after_key(const std::string& json, size_t search_from, const char* key, bool& value) {
    const size_t key_pos = json.find(key, search_from);
    if (key_pos == std::string::npos) {
      return true;
    }
    const size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) {
      return false;
    }
    const size_t val = skip_ws(json, colon + 1);
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

  static bool parse_int_after_key(const std::string& json, size_t search_from, const char* key, int& value) {
    const size_t key_pos = json.find(key, search_from);
    if (key_pos == std::string::npos) {
      return true;
    }
    const size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) {
      return false;
    }
    size_t pos = skip_ws(json, colon + 1);
    bool neg = false;
    if (pos < json.size() && json[pos] == '-') {
      neg = true;
      ++pos;
    }
    int parsed = 0;
    bool saw_digit = false;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
      saw_digit = true;
      parsed = parsed * 10 + (json[pos] - '0');
      ++pos;
    }
    if (!saw_digit) {
      return false;
    }
    value = neg ? -parsed : parsed;
    return true;
  }

  static bool parse_metadata_json(const std::string& json, TensorRTBlobHeader& out) {
    out.input_binding_names.clear();
    out.output_binding_names.clear();
    out.hardware_compatible = false;
    out.device_id = 0;

    const size_t bindings_pos = json.find("\"io_bindings\"");
    if (bindings_pos == std::string::npos) {
      return false;
    }
    const size_t arr_start = json.find('[', bindings_pos);
    if (arr_start == std::string::npos) {
      return false;
    }

    size_t pos = arr_start + 1;
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
      bool saw_name = false;

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
          saw_name = pos != std::string::npos;
          if (!saw_name) {
            return false;
          }
        } else if (key == "is_input") {
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

      if (saw_name && !name.empty()) {
        if (is_input) {
          out.input_binding_names.push_back(name);
        } else {
          out.output_binding_names.push_back(name);
        }
      }
    }

    return parse_bool_after_key(json, pos, "\"hardware_compatible\"", out.hardware_compatible) &&
        parse_int_after_key(json, pos, "\"device_id\"", out.device_id);
  }
};

} // namespace executorch_backend
} // namespace torch_tensorrt
