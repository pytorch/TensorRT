#include "torch/csrc/jit/passes/dead_code_elimination.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

// Schemas for dropout variants
const std::unordered_set<c10::Symbol> DropoutNodeKinds = {
    c10::Symbol::fromQualString("aten::dropout"),
    c10::Symbol::fromQualString("aten::dropout_"),
    c10::Symbol::fromQualString("aten::feature_dropout"),
    c10::Symbol::fromQualString("aten::feature_dropout_"),
    c10::Symbol::fromQualString("aten::feature_alpha_dropout"),
    c10::Symbol::fromQualString("aten::feature_alpha_dropout_"),
};

void removeDropoutInBlock(torch::jit::Block* block) {
  /*
  Function adapted from:
  torch/csrc/jit/passes/remove_dropout.cpp

  Modified for conciseness, documentation, and allowing new variants of dropout operators to be quickly added
  */
  std::vector<torch::jit::Node*> dropout_nodes_to_remove;

  for (auto node : block->nodes()) {
    // Remove dropout for each member block within a node
    for (auto block : node->blocks()) {
      removeDropoutInBlock(block);
    }

    // For each node having a dropout-variant Schema, remove the node
    if (DropoutNodeKinds.find(node->kind()) != DropoutNodeKinds.end()) {
      // Extract input and output tensors of dropout operator
      auto input_value = node->inputs()[0];
      auto output_value = node->outputs()[0];

      output_value->replaceAllUsesWith(input_value);
      dropout_nodes_to_remove.push_back(node);
    }
  }

  // Delete dropout nodes
  for (auto del_node : dropout_nodes_to_remove) {
    del_node->destroy();
  }
}

void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph) {
  // Remove all instances of dropout variants from graph
  removeDropoutInBlock(graph->block());
  torch::jit::EliminateDeadCode(graph);
  LOG_GRAPH("Post remove dropout: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
