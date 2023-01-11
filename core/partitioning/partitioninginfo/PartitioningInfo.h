#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

struct PartitioningInfo {
  ir::CollectionInputSpecMap collection_input_spec_map;
  bool enabled = false;
  uint64_t min_block_size = 1;
  std::vector<std::string> forced_fallback_operators;
  bool truncate_long_and_double;
  ir::Device target_device;
  bool cast_int8_inputs = false;

  std::string getGPUDeviceString() const {
    return "cuda:" + std::to_string(target_device.gpu_id);
  };
};

std::ostream& operator<<(std::ostream& os, const PartitioningInfo& s);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
