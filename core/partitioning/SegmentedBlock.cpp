#include "SegmentedBlock.h"

namespace trtorch {
namespace core {
namespace partitioning {

torch::jit::Value* SegmentedBlock::getOrAddInputForValue(torch::jit::Value* old_value) {
  if (old_to_new_.count(old_value) == 0) {
    auto node = old_value->node();

    if (node->kind() == torch::jit::prim::Constant) {
      auto new_const = g_->createClone(node, {nullptr});
      g_->block()->prependNode(new_const);
      return new_const->output();
    }
    auto new_value = g_->block()->addInput();
    // every time when we addInput, we push back the corresponding lowering graph torch::jit::Value to our raw_inputs
    inputs_.push_back(old_value);
    old_to_new_[old_value] = new_value;
    new_value->copyMetadata(old_value);
    return new_value;
  } else {
    return old_to_new_[old_value];
  }
}

torch::jit::Node* SegmentedBlock::cloneNode(torch::jit::Node* node) {
  auto* block = g_->block();
  auto env = [&](torch::jit::Value* v) { return getOrAddInputForValue(v); };

  // create node for current graph by using the metadata in node and input Values in env
  auto new_node = block->appendNode(g_->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new_[oo] = no;
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