#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace partitioning {

struct PartitionInfo {
  bool enabled = false;
  uint64_t min_block_size = 1;
  std::vector<std::string> forced_fallback_operators;
  bool truncate_long_and_double;
};

std::ostream& operator<<(std::ostream& os, const PartitionInfo& s);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
