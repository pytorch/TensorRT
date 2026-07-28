#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

struct TensorRTBlobHeader {
  uint32_t metadata_offset = 0;
  uint32_t metadata_size = 0;
  uint32_t engine_offset = 0;
  uint64_t engine_size = 0;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  bool hardware_compatible = false;
  int device_id = 0;

  static const void* engine_data(const void* blob, const TensorRTBlobHeader& h);
  static bool parse(const void* data, std::size_t size, TensorRTBlobHeader& out);
};

} // namespace executorch_backend
} // namespace torch_tensorrt
