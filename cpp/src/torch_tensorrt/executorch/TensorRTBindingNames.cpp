#include "torch_tensorrt/executorch/TensorRTBindingNames.h"

#include <charconv>
#include <cstddef>
#include <string>
#include <system_error>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace detail {

bool parse_binding_index(const std::string& name, std::size_t& index) {
  const std::size_t delim = name.find_last_of("._");
  if (delim == std::string::npos) {
    return false;
  }
  const std::size_t index_start = delim + 1;
  if (index_start >= name.size()) {
    return false;
  }

  const char* begin = name.data() + index_start;
  const char* end = name.data() + name.size();
  const auto result = std::from_chars(begin, end, index);
  return result.ec == std::errc() && result.ptr == end;
}

bool append_binding_name(std::vector<std::string>& names, const std::string& name) {
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

bool all_binding_names_present(const std::vector<std::string>& names) {
  for (const auto& name : names) {
    if (name.empty()) {
      return false;
    }
  }
  return true;
}

} // namespace detail
} // namespace executorch_backend
} // namespace torch_tensorrt
