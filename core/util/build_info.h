#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include "ATen/Version.h"
#include "NvInferVersion.h"

namespace torch_tensorrt {
namespace core {
namespace util {
inline std::string get_build_info() {
  std::stringstream info;
  info << "Using TensorRT Version: " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '.'
       << NV_TENSORRT_BUILD << '\n'
       << at::show_config();
  return info.str();
}
} // namespace util
} // namespace core
} // namespace torch_tensorrt
