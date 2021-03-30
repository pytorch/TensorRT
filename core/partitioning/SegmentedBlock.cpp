#include "SegmentedBlock.h"

namespace trtorch {
namespace core {
namespace partitioning {

torch::jit::Value* getOrAddInputForValue(
    torch::jit::Value* old_value,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
  if (old_to_new.count(old_value) == 0) {
    auto node = old_value->node();

    if (node->kind() == torch::jit::prim::Constant) {
      auto new_const = graph->createClone(node, {nullptr});
      graph->block()->prependNode(new_const);
      return new_const->output();
    }
    auto new_value = graph->block()->addInput();
    old_to_new[old_value] = new_value;
    new_value->copyMetadata(old_value);
    // mapping from new graph input Values to original graph values
    old_to_new[new_value] = old_value;
    return new_value;
  } else {
    return old_to_new[old_value];
  }
}

torch::jit::Node* cloneNode(
    torch::jit::Node* node,
    std::shared_ptr<torch::jit::Graph>& graph,
    std::unordered_map<torch::jit::Value*, torch::jit::Value*>& old_to_new) {
  auto* block = graph->block();
  auto env = [&](torch::jit::Value* v) { return getOrAddInputForValue(v, graph, old_to_new); };

  // create node for current graph by using the metadata in node and input Values in env
  auto new_node = block->appendNode(graph->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new[oo] = no;
  }
  return new_node;
}

std::vector<SegmentedBlock> segment_graph(
    std::shared_ptr<torch::jit::Graph> g,
    const conversion::TorchFallback& fallback_info) {
  auto min_block_size = fallback_info.min_block_size;
  std::unordered_set<std::string> forced_fallback_operators(
      fallback_info.forced_fallback_operators.begin(), fallback_info.forced_fallback_operators.end());

  auto nodes = g->block()->nodes();
  std::vector<SegmentedBlock> segmented_blocks;

  // segment the nodes
  std::vector<torch::jit::Node*> tensorrt_nodes, pytorch_nodes;
  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant)
      continue;

    std::string node_string(n->kind().toQualString());
    if (conversion::OpSupported(n) && !forced_fallback_operators.count(node_string)) {
      tensorrt_nodes.push_back(n);
      if (tensorrt_nodes.size() >= min_block_size && !pytorch_nodes.empty()) {
        segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
        pytorch_nodes.clear();
      }
    } else {
      if (tensorrt_nodes.size() >= min_block_size) {
        segmented_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
      } else {
        pytorch_nodes.insert(pytorch_nodes.end(), tensorrt_nodes.begin(), tensorrt_nodes.end());
      }
      tensorrt_nodes.clear();
      pytorch_nodes.push_back(n);
    }
  }

  // if there is any kTorch nodes left, then either the last nodes are kTorch or last nodes are kTensorRT but num <
  // min_block_size
  if (!pytorch_nodes.empty()) {
    pytorch_nodes.insert(pytorch_nodes.end(), tensorrt_nodes.begin(), tensorrt_nodes.end());
    segmented_blocks.emplace_back(SegmentedBlock::kTorch, pytorch_nodes);
  } else {
    segmented_blocks.emplace_back(SegmentedBlock::kTensorRT, tensorrt_nodes);
  }

  return std::move(segmented_blocks);
}

} // namespace partitioning
} // namespace core
} // namespace trtorch