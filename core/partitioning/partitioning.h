#pragma once

#include <iostream>
#include <vector>

#include "core/ir/ir.h"
#include "core/partitioning/PartitionInfo.h"
#include "core/partitioning/SegmentedBlock.h"
#include "core/partitioning/shape_analysis.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

typedef std::vector<SegmentedBlock> PartitionedGraph;

PartitionedGraph segment_graph(
    torch::jit::Block* block,
    const PartitionInfo& partition_info,
    std::unordered_map<torch::jit::Node*, int>& fallback_nodes);

PartitionedGraph Partition(
    torch::jit::Block* block,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& example_tensor_map,
    const PartitionInfo& partition_info,
    std::unordered_map<torch::jit::Node*, int>& fallback_nodes);

std::ostream& operator<<(std::ostream& os, const PartitionedGraph& g);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
