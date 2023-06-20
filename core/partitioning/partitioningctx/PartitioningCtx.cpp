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
  _load_nodes_into_decision_map(b);
}

void PartitioningCtx::_load_nodes_into_decision_map(torch::jit::Block* b) {
  if (b->owningNode() && b->owningNode()->kind() == torch::jit::prim::Loop)
    return;

  original_blocks.push_back(b);

  for (const auto n : b->nodes()) {
    if (n->kind() == torch::jit::prim::Constant) {
      continue;
    }
    node_executor_decision_map[n] = NodeExecutorDecision::kUNKNOWN;
    for (const auto sub_b : n->blocks()) {
      _load_nodes_into_decision_map(sub_b);
    }
  }
}

void PartitioningCtx::setNodeExecutorDecision(torch::jit::Node* n, NodeExecutorDecision decision) {
  auto iter = node_executor_decision_map.find(n);
  auto prev_decision = NodeExecutorDecision::kUNKNOWN;
  if (iter != node_executor_decision_map.end()) {
    prev_decision = iter->second;
  }
  LOG_DEBUG("Setting node " << util::node_info(n) << " " << decision << " (previously was " << prev_decision << ")");

  node_executor_decision_map[n] = decision;
  return;
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

std::vector<torch::jit::Node*> PartitioningCtx::getNodesRunInTorch() {
  std::vector<torch::jit::Node*> nodes_run_in_torch;
  for (auto i : node_executor_decision_map) {
    if (i.second != NodeExecutorDecision::kCONVERT) {
      nodes_run_in_torch.push_back(i.first);
    }
  }
  return nodes_run_in_torch;
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
