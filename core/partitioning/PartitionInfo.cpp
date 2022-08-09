#include <iostream>
#include <sstream>
#include <utility>

#include "core/partitioning/PartitionInfo.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {
// clang-format off
std::ostream& operator<<(std::ostream& os, const PartitionInfo& s) {
  os << "Settings requested for Torch Fallback:" \
     << "\n    \"enabled\": ";
  if (s.enabled) {
    os << "True";
    os << "\n    \"min_block_size\": " << s.min_block_size \
       << "\n    \"torch_executed_operators\": [";
    for (auto i : s.forced_fallback_operators) {
      os <<"\n        " << i << ',';
    }
    os << "\n     ]";
  } else {
    os << "False";
  }
  return os;
}
// clang-format on
} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
