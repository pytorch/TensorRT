#include <queue>

#include "core/partitioning/partitioningctx/PartitioningCtx.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace partitioning {

PartitioningCtx::PartitioningCtx(torch::jit::Block* b, PartitioningInfo info)
    : settings(info),
      forced_fallback_ops(info.forced_fallback_operators.begin(), info.forced_fallback_operators.end()) {
  LOG_DEBUG(settings);
  //_load_nodes_into_decision_map(b);
}

void PartitioningCtx::_load_nodes_into_decision_map(torch::jit::Block* b) {
  for (const auto n : b->nodes()) {
    node_executor_decision_map[n] = NodeExecutorDecision::kUNKNOWN;
    for (const auto sub_b : n->blocks()) {
      _load_nodes_into_decision_map(sub_b);
    }
  }
}

void PartitioningCtx::finalizeNewBlock(
    SegmentedBlock::SegmentedBlockTarget kind,
    std::vector<torch::jit::Node*>& nodes) {
  LOG_DEBUG("Finalizing in progress " << SegmentedBlock::target_to_str(kind) << " block");
  blocks.emplace_back(blocks.size(), kind, nodes);

  // TODO: Can we not need this?
  nodes.clear();
  LOG_DEBUG(blocks.back());
}

bool PartitioningCtx::setNodeExecutorDecision(torch::jit::Node* n, NodeExecutorDecision decision) {
  auto iter = node_executor_decision_map.find(n);
  auto prev_decision = NodeExecutorDecision::kUNKNOWN;
  if (iter != node_executor_decision_map.end()) {
    prev_decision = iter->second;
  }
  LOG_GRAPH("Setting node " << util::node_info(n) << " " << decision << " (previously was " << prev_decision << ")");

  // NOTE: This is this way due to partitioning.cpp L#134 I dont know if this is what we should do.
  auto result = node_executor_decision_map.insert({n, decision});
  return result.second;
}

bool PartitioningCtx::shouldNodeRunInTorch(torch::jit::Node* n) {
  auto iter = node_executor_decision_map.find(n);
  auto decision = NodeExecutorDecision::kUNKNOWN;
  if (iter != node_executor_decision_map.end()) {
    decision = iter->second;
  }

  if (decision == NodeExecutorDecision::kCONVERT || decision == NodeExecutorDecision::kUNKNOWN) {
    return false;
  } else {
    return true;
  }
}

bool PartitioningCtx::shouldNodeRunInTensorRT(torch::jit::Node* n) {
  auto iter = node_executor_decision_map.find(n);
  auto decision = NodeExecutorDecision::kUNKNOWN;
  if (iter != node_executor_decision_map.end()) {
    decision = iter->second;
  }

  if (decision == NodeExecutorDecision::kCONVERT) {
    return true;
  } else {
    return false;
  }
}

bool PartitioningCtx::isNodeExecutorKnown(torch::jit::Node* n) {
  auto iter = node_executor_decision_map.find(n);
  auto decision = NodeExecutorDecision::kUNKNOWN;
  if (iter != node_executor_decision_map.end()) {
    decision = iter->second;
  }

  if (decision == NodeExecutorDecision::kUNKNOWN) {
    return false;
  } else {
    return true;
  }
}

std::ostream& operator<<(std::ostream& os, const NodeExecutorDecision& format) {
  switch (format) {
    case NodeExecutorDecision::kUNSUPPORTED:
      return os << "to run torch due to lack of converter support";
    case NodeExecutorDecision::kOPERATOR_FALLBACK:
      return os << "to run torch due to user expectily requesting op kind runs in torch";
    case NodeExecutorDecision::kMODULE_FALLBACK:
      return os << "to run torch due to being a member of a module user has requested to run in torch";
    case NodeExecutorDecision::kMIN_BLOCK_FALLBACK:
      return os << "to run torch due owning block not large enough to exceed user specified min_block_size";
    case NodeExecutorDecision::kNON_TENSOR:
      return os << "to run torch due to producing or consuming non-tensor values";
    case NodeExecutorDecision::kCONVERT:
      return os << "to run in tensorrt";
    case NodeExecutorDecision::kUNKNOWN:
    default:
      return os << "unknown node executor decision";
  }
}

std::ostream& operator<<(std::ostream& os, const PartitionedGraph& g) {
  os << "Partitioned Graph: [";
  for (auto b : g) {
    os << b;
  }
  os << "]";
  return os;
}

} // namespace partitioning
} // namespace core
} // namespace torch_tensorrt
