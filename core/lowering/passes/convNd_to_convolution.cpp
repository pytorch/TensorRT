#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "torch/csrc/jit/ir/irparser.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void replaceConv(
    torch::jit::Block* block,
    const std::string& node_kind,
    const std::string& unwrapped_conv,
    const size_t num_input_args) {
  // Iterate through nodes in block, seaching for aten::conv*
  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;

    // Recursively explore nested blocks, such as those arising from prim::If
    for (auto nested_block : n->blocks()) {
      replaceConv(nested_block, node_kind, unwrapped_conv, num_input_args);
    }

    // If node matches desired kind and number of input arguments, replace it
    if ((n->kind().toQualString() == node_kind) && (n->inputs().size() == num_input_args)) {
      // Establish insert point within block
      torch::jit::WithInsertPoint guard(*it);

      // Initialize new fused subgraph from IR code provided
      auto fused_g = std::make_shared<torch::jit::Graph>();
      torch::jit::parseIR(unwrapped_conv, fused_g.get());

      // Insert subgraph in place of aten::conv*, replacing inputs and outputs accordingly
      torch::jit::Value* new_output = insertGraph(*it->owningGraph(), *fused_g, it->inputs()).at(0);
      new_output->setType(it->output()->type());
      it->output()->replaceAllUsesWith(new_output);
      it.destroyCurrent();
    }
  }
}

void Conv1DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  const std::string conv1d_node_kind = "aten::conv1d";
  const std::string convolution_pattern = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %1 : bool = prim::Constant[value=0]()
            %2 : int[] = prim::Constant[value=[0]]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %2, %g, %1, %1, %1, %1)
            return (%4))IR";

  // Schema is aten::conv1d(%x, %w, %b, %s, %p, %d, %g) --> 7 inputs
  replaceConv(graph->block(), conv1d_node_kind, convolution_pattern, 7);
  LOG_GRAPH("Post map conv1d -> _convolution: " << *graph);
}

void ConvTransposed1DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  const std::string conv_transpose1d_node_kind = "aten::conv_transpose1d";
  const std::string convolution_pattern = R"IR(
        graph(%x, %w, %b, %s, %p, %o, %g, %d):
            %1 : bool = prim::Constant[value=1]()
            %2 : bool = prim::Constant[value=1]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %o, %g, %2, %2, %2, %2)
            return (%4))IR";

  // Schema is aten::conv_transpose1d(%x, %w, %b, %s, %p, %o, %g, %d) --> 8 inputs
  replaceConv(graph->block(), conv_transpose1d_node_kind, convolution_pattern, 8);
  LOG_GRAPH("Post map conv_transpose1d -> _convolution: " << *graph);
}

void Conv2DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  const std::string conv2d_node_kind = "aten::conv2d";
  const std::string convolution_pattern = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %1 : bool = prim::Constant[value=0]()
            %2 : int[] = prim::Constant[value=[0, 0]]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %2, %g, %1, %1, %1, %1)
            return (%4))IR";

  // Schema is aten::conv2d(%x, %w, %b, %s, %p, %d, %g) --> 7 inputs
  replaceConv(graph->block(), conv2d_node_kind, convolution_pattern, 7);
  LOG_GRAPH("Post map conv2d -> _convolution: " << *graph);
}

void Conv3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  const std::string conv3d_node_kind = "aten::conv3d";
  const std::string convolution_pattern = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %1 : bool = prim::Constant[value=0]()
            %2 : int[] = prim::Constant[value=[0, 0, 0]]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %2, %g, %1, %1, %1, %1)
            return (%4))IR";

  // Schema is aten::conv3d(%x, %w, %b, %s, %p, %d, %g) --> 7 inputs
  replaceConv(graph->block(), conv3d_node_kind, convolution_pattern, 7);
  LOG_GRAPH("Post map conv3d -> _convolution: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
