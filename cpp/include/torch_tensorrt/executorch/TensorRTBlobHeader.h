#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

// One aliased output->input binding pair (KV-cache in-place update, or a
// user-declared alias). The engine's output binding shares device memory with
// the named input binding; the runtime binds the output to the input's tensor
// so the update lands in-place in the caller-owned buffer.
struct AliasedBinding {
  std::string output; // output binding name
  std::string input; // input binding name it aliases
  std::string kind; // "kv_cache_update" (TRT-enforced) or "user"
};

struct TensorRTBlobHeader {
  uint32_t metadata_offset = 0;
  uint32_t metadata_size = 0;
  uint32_t engine_offset = 0;
  uint64_t engine_size = 0;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  std::vector<AliasedBinding> aliased_io;
  bool hardware_compatible = false;
  int device_id = 0;

  static const void* engine_data(const void* blob, const TensorRTBlobHeader& h);
  static bool parse(const void* data, std::size_t size, TensorRTBlobHeader& out);
};

} // namespace executorch_backend
} // namespace torch_tensorrt
