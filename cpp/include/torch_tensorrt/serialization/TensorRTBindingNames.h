#pragma once

#include "NvInfer.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch_tensorrt {
namespace serialization {

inline constexpr char kBindingNameDelimiter = '%';

struct TensorRTBindingNames {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::unordered_map<std::uint64_t, std::uint64_t> input_map;
  std::unordered_map<std::uint64_t, std::uint64_t> output_map;
};

inline std::string serialize_binding_names(const std::vector<std::string>& bindings) {
  std::string serialized;
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    if (i > 0) {
      serialized.push_back(kBindingNameDelimiter);
    }
    serialized += bindings[i];
  }
  return serialized;
}

inline std::vector<std::string> split_serialized_binding_names(const std::string& serialized) {
  std::vector<std::string> bindings;
  std::size_t start = 0;
  while (start < serialized.size()) {
    std::size_t end = serialized.find(kBindingNameDelimiter, start);
    if (end == std::string::npos) {
      end = serialized.size();
    }
    if (end > start) {
      bindings.emplace_back(serialized.substr(start, end - start));
    }
    start = end + 1;
  }
  return bindings;
}

inline bool parse_binding_index(const std::string& name, std::size_t& index) {
  const std::size_t delim = name.find_last_of("._");
  if (delim == std::string::npos) {
    return false;
  }
  const std::size_t index_start = delim + 1;
  if (index_start >= name.size()) {
    return false;
  }

  std::size_t parsed_index = 0;
  const char* begin = name.data() + index_start;
  const char* end = name.data() + name.size();
  const auto result = std::from_chars(begin, end, parsed_index);
  if (result.ec != std::errc() || result.ptr != end) {
    return false;
  }

  index = parsed_index;
  return true;
}

inline bool append_binding_name(std::vector<std::string>& names, const std::string& name) {
  std::size_t position = 0;
  if (!parse_binding_index(name, position)) {
    return false;
  }

  if (names.size() <= position) {
    names.resize(position + 1);
  }
  if (!names[position].empty()) {
    return false;
  }
  names[position] = name;
  return true;
}

inline bool all_binding_names_present(const std::vector<std::string>& names) {
  for (const auto& name : names) {
    if (name.empty()) {
      return false;
    }
  }
  return true;
}

inline bool infer_engine_binding_names(const nvinfer1::ICudaEngine& engine, TensorRTBindingNames& out) {
  TensorRTBindingNames inferred;

  for (std::int32_t trt_idx = 0; trt_idx < engine.getNbIOTensors(); ++trt_idx) {
    const char* raw_name = engine.getIOTensorName(trt_idx);
    if (raw_name == nullptr) {
      return false;
    }

    const std::string name(raw_name);
    std::size_t pyt_idx = 0;
    if (!parse_binding_index(name, pyt_idx)) {
      return false;
    }

    const bool is_input = engine.getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
    auto& target_names = is_input ? inferred.input_names : inferred.output_names;
    auto& target_map = is_input ? inferred.input_map : inferred.output_map;
    if (!append_binding_name(target_names, name)) {
      return false;
    }
    target_map[static_cast<std::uint64_t>(trt_idx)] = static_cast<std::uint64_t>(pyt_idx);
  }

  if (!all_binding_names_present(inferred.input_names) || !all_binding_names_present(inferred.output_names)) {
    return false;
  }

  out = std::move(inferred);
  return true;
}

} // namespace serialization
} // namespace torch_tensorrt
