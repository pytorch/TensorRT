#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace torch_tensorrt {
namespace executorch_backend {

struct EdgeLLMBlobHeader {
  uint32_t metadata_offset = 0;
  uint32_t metadata_size = 0;
  uint32_t blob_offset = 0;
  uint64_t blob_size = 0;
  int abi_version = 0;
  std::string component;
  std::string runner;
  std::string metadata_json;
  std::string runner_config_json;

  static const void* nested_blob_data(const void* payload, const EdgeLLMBlobHeader& h);
  static bool parse(const void* data, std::size_t size, EdgeLLMBlobHeader& out);
};

} // namespace executorch_backend
} // namespace torch_tensorrt
