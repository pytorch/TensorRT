#pragma once

#include <vector>

#include "core/ir/ir.h"
#include "core/partitioning/PartitionInfo.h"
#include "core/partitioning/SegmentedBlock.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

typedef std::vector<SegmentedBlock> PartitionedGraph;

PartitionedGraph segment_graph(std::shared_ptr<torch::jit::Graph> g, const PartitionInfo& partition_info);

std::vector<SegmentedBlock> Partition(
    std::shared_ptr<torch::jit::Graph> g,
    std::vector<ir::Input>& inputs,
    const PartitionInfo& partition_info);

} // namespace partitioning
} // namespace core
} // namespace trtorch