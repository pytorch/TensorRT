#pragma once

#include <iostream>
#include <vector>

#include "torch/csrc/jit/ir/ir.h"

#include "core/ir/ir.h"
#include "core/partitioning/partitioninginfo/PartitioningInfo.h"
#include "core/partitioning/segmentedblock/SegmentedBlock.h"
#include "core/util/prelude.h"

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

std::unordered_map<const torch::jit::Value*, torch::jit::IValue> generateRandomInputs(
    std::unordered_map<const torch::jit::Value*, std::vector<ir::Input>>& input_ranges,
    std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>>& input_types);

void runShapeAnalysis(
    std::vector<SegmentedBlock>& segmented_blocks,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& ivalues_maps,
    const PartitioningInfo& partitioning_info);

PartitionedGraph segment_graph(
    torch::jit::Block* block,
    const PartitioningInfo& partitioning_info,
    std::unordered_map<torch::jit::Node*, int>& fallback_nodes);

PartitionedGraph Partition(
    torch::jit::Block* block,
    std::unordered_map<const torch::jit::Value*, torch::jit::IValue>& example_tensor_map,
    const PartitioningInfo& partitioning_info,
    std::unordered_map<torch::jit::Node*, int>& fallback_nodes);

std::ostream& operator<<(std::ostream& os, const PartitionedGraph& g);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
