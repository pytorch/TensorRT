#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace trtorch {
namespace core {
namespace partitioning {

struct PartitionInfo {
    bool enabled = false;
    uint64_t min_block_size = 1;
    std::vector<std::string> forced_fallback_operators;
};

} // namespace partitioning
} // namespace core
} // namespace trtorch