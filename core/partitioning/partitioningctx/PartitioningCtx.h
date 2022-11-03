#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/partitioning/partitioninginfo/PartitioningInfo.h"
#include "core/partitioning/segmentedblock/SegmentedBlock.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

enum NodeExecutorDecision {
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
  /// This node is going to be converted
  kCONVERT,
  /// Sentinel
  kUNKNOWN,
};

std::ostream& operator<<(std::ostream& os, const NodeExecutorDecision& format);

typedef std::unordered_map<torch::jit::Node*, NodeExecutorDecision> NodeExecutorDecisionMap;

typedef std::vector<SegmentedBlock> PartitionedGraph;

std::ostream& operator<<(std::ostream& os, const PartitionedGraph& g);

struct UsageInfo {
  size_t produce_id; // id of segmented block which contains a raw value of a given torch::jit::Value
  std::vector<size_t> torch_use_id; // ids of segmented blocks which are of type Pytorch
  std::vector<size_t> tensorrt_use_id; // ids of segmented blocks which are of type TensorRT
};

struct PartitioningCtx {
  // TODO: Make the set a part of settings not stand alone
  PartitioningInfo settings;
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> min_input_ivalues_map;
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> opt_input_ivalues_map;
  std::unordered_map<const torch::jit::Value*, torch::jit::IValue> max_input_ivalues_map;
  // records all the original blocks topologically in the module
  std::vector<torch::jit::Block*> original_blocks;
  // mapping: node=> execution status
  NodeExecutorDecisionMap node_executor_decision_map;
  // LUT of the segmented blocks for each blocks in the module
  std::unordered_map<torch::jit::Block*, PartitionedGraph> partitioned_blocks;
  std::unordered_set<std::string> forced_fallback_ops;

  PartitioningCtx(torch::jit::Block* b, PartitioningInfo info);
  void setNodeExecutorDecision(torch::jit::Node* n, NodeExecutorDecision decision);
  bool shouldNodeRunInTorch(torch::jit::Node* n);
  bool shouldNodeRunInTensorRT(torch::jit::Node* n);
  std::vector<torch::jit::Node*> getNodesRunInTorch();
  std::unordered_map<const torch::jit::Value*, std::vector<c10::optional<at::ScalarType>>> input_types_map;

 private:
  void _load_nodes_into_decision_map(torch::jit::Block* b);
};

std::ostream& operator<<(std::ostream& os, const PartitioningCtx& s);

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
