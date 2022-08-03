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

enum FallbackNodeType {
  /// Node is not supported by TensorRT
  kUNSUPPORTED,
  /// Node is explicitly forced to fallback to Pytorch due to operator fallback
  kOPERATOR_FALLBACK,
  /// Node is explicitly forced to fallback to Pytorch due to module fallback
  kMODULE_FALLBACK,
  /// This node is in a TRT segment which does not satisfy min_block_size
  /// and hence is forced to fallback.
  kMIN_BLOCK_FALLBACK,
  /// This node produces/consumes non-tensor inputs
  kNON_TENSOR,
};

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
