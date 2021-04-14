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
    torch::jit::Block* block,
    std::vector<ir::InputRange>& input_ranges,
    const PartitionInfo& partition_info);



} // namespace partitioning
} // namespace core
} // namespace trtorch
