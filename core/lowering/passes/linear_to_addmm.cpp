
#include <torch/csrc/jit/runtime/operator.h>
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/function_impl.h"
#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/ir/irparser.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void replaceLinear(torch::jit::Block* block) {
  // Define the decomposition function for aten::linear for the case where bias (mat2) is None.
  static torch::jit::CompilationUnit decompose_funcs(R"SCRIPT(
     def linear(self: Tensor, mat1: Tensor, mat2: Tensor):
         return torch.matmul(self, mat1.t())
     )SCRIPT");

  // Define graph format for aten::linear with Tensor-type bias
  std::string fused_linear = R"IR(
        graph(%input, %weight, %bias):
            %1: int = prim::Constant[value=1]()
            %weight = aten::t(%weight)
            %mm: Tensor = aten::matmul(%input, %weight)
            %b_f: Tensor = trt::const(%bias)
            %out: Tensor = aten::add(%b_f, %mm, %1)
            return (%out))IR";

  // Iterate through nodes in block, seaching for aten::linear
  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;

    // Recursively explore nested blocks, such as those arising from prim::If
    for (auto block : n->blocks()) {
      replaceLinear(block);
    }

    if ((n->kind().toQualString() == std::string("aten::linear")) && (n->inputs().size() >= 3)) {
      auto input_values = n->inputs();

      // input_values[2] is the bias
      // If Tensor, replace with fused-bias decomposed graph
      // Otherwise, replace it with the no-bias decomposed linear graph.
      if (input_values[2]->type()->isSubtypeOf(c10::TensorType::get())) {
        torch::jit::WithInsertPoint guard(*it);

        // Initialize new fused subgraph from IR code above
        auto fused_g = std::make_shared<torch::jit::Graph>();
        torch::jit::parseIR(fused_linear, fused_g.get());

        // Insert subgraph in place of aten::linear, replacing inputs and outputs accordingly
        torch::jit::Value* new_output = insertGraph(*it->owningGraph(), *fused_g, it->inputs()).at(0);
        new_output->setType(it->output()->type());
        it->output()->replaceAllUsesWith(new_output);
        it.destroyCurrent();
      } else {
        torch::jit::WithInsertPoint guard(*it);

        // Initialized decomposed graph without bias term
        std::shared_ptr<torch::jit::Graph> d_graph = toGraphFunction(decompose_funcs.get_function("linear")).graph();
        torch::jit::Value* new_output = insertGraph(*it->owningGraph(), *d_graph, it->inputs()).at(0);

        // Insert function in place of aten::linear, replacing inputs and outputs accordingly
        new_output->setType(it->output()->type());
        it->output()->replaceAllUsesWith(new_output);
        it.destroyCurrent();
      }
    }
  }
}

void LinearToAddMM(std::shared_ptr<torch::jit::Graph>& graph) {
  // Recursively find and replace all instances of aten::linear with the corresponding decomposed form
  replaceLinear(graph->block());
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
