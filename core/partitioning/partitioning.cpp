#include "partitioning.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace partitioning {

torch::jit::Value* getOrAddInputForValue(torch::jit::Value* old_value, std::shared_ptr<torch::jit::Graph> &graph,
                                         std::unordered_map<torch::jit::Value*, torch::jit::Value*> &old_to_new) {
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
    return new_value;
  } else {
    return old_to_new[old_value];
  }
}

torch::jit::Node* cloneNode(torch::jit::Node* node, std::shared_ptr<torch::jit::Graph> &graph,
                            std::unordered_map<torch::jit::Value*, torch::jit::Value*> &old_to_new) {
  auto* block = graph->block();
  auto env = [&](torch::jit::Value* v) { return getOrAddInputForValue(v, graph, old_to_new); };

  auto new_node = block->appendNode(graph->createClone(node, env));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    old_to_new[oo] = no;
  }

  return new_node;
}


std::vector<SegmentedBlock> segment_graph(std::shared_ptr<torch::jit::Graph> g) {
  std::vector<SegmentedBlock> segmented_blocks;

  auto nodes = g->block()->nodes();

  for (const auto n : nodes) {
    if (n->kind() == torch::jit::prim::Constant) continue;
    auto block_target = conversion::OpSupported(n) ? SegmentedBlock::kTensorRT : SegmentedBlock::kTorch;

    if (segmented_blocks.empty() || block_target != segmented_blocks.back().target) {
      SegmentedBlock cur_block(block_target);
      cur_block.appendNode(n);
      segmented_blocks.push_back(cur_block);
    } else {
        segmented_blocks.back().appendNode(n);
    }
  }

  for (auto &seg_block : segmented_blocks) {
    seg_block.registerOutput();
  }

  return segmented_blocks;
}

}
}
}


