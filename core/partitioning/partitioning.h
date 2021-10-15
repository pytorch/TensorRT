#pragma once

#include <vector>
#include <iostream>

#include "core/ir/ir.h"
#include "core/partitioning/PartitionInfo.h"
#include "core/partitioning/SegmentedBlock.h"
#include "core/partitioning/shape_analysis.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace partitioning {

typedef std::vector<SegmentedBlock> PartitionedGraph;

PartitionedGraph segment_graph(torch::jit::Block* block, const PartitionInfo& partition_info);

PartitionedGraph Partition(
    torch::jit::Block* block,
    std::unordered_map<torch::jit::Value*, torch::jit::IValue>& input_ivalues_map,
    const PartitionInfo& partition_info);

std::ostream& operator<<(std::ostream& os, const PartitionedGraph& g);

} // namespace partitioning
} // namespace core
} // namespace trtorch
