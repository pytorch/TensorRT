#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace detail {

bool parse_binding_index(const std::string& name, std::size_t& index);
bool append_binding_name(std::vector<std::string>& names, const std::string& name);
bool all_binding_names_present(const std::vector<std::string>& names);

} // namespace detail
} // namespace executorch_backend
} // namespace torch_tensorrt
