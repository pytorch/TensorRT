
#include <torch/csrc/jit/runtime/operator.h>
#include "core/util/prelude.h"
#include "torch/csrc/jit/api/function_impl.h"
#include "torch/csrc/jit/ir/alias_analysis.h"
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

void replaceLinearWithBiasNonePattern(std::shared_ptr<torch::jit::Graph> graph) {
  // Define the decomposition function for aten::linear for the case where bias (mat2) is None.
  static torch::jit::CompilationUnit decompose_funcs(R"SCRIPT(
     def linear(self: Tensor, mat1: Tensor, mat2: Tensor):
         return torch.matmul(self, mat1.t())
     )SCRIPT");

  // Iterate through nodes and search for aten::linear nodes where bias is not a Tensor (includes bias=None case)
  auto block = graph->block();
  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    auto n = *it;
    if (n->kind().toQualString() == std::string("aten::linear")) {
      auto input_values = n->inputs();
      // input_values[2] is the bias. If none, replace it with the decomposed linear graph.
      if (input_values[2]->type()->isSubtypeOf(c10::TensorType::get())) {
        continue;
      } else {
        torch::jit::WithInsertPoint guard(*it);
        std::shared_ptr<torch::jit::Graph> d_graph = toGraphFunction(decompose_funcs.get_function("linear")).graph();
        torch::jit::Value* new_output = insertGraph(*it->owningGraph(), *d_graph, it->inputs()).at(0);
        new_output->setType(it->output()->type());
        it->output()->replaceAllUsesWith(new_output);
        it.destroyCurrent();
      }
    }
  }
}

void LinearToAddMM(std::shared_ptr<torch::jit::Graph>& graph) {
  // TensorRT implicitly adds a flatten layer infront of FC layers if necessary
  std::string flatten_linear_pattern = R"IR(
        graph(%input, %weight, %bias):
            %res = aten::linear(%input, %weight, %bias)
            return (%res))IR";

  std::string fused_linear = R"IR(
        graph(%input, %weight_t, %bias):
            %1: int = prim::Constant[value=1]()
            %weight = aten::t(%weight_t)
            %mm: Tensor = aten::matmul(%input, %weight)
            %b_f: Tensor = trt::const(%bias)
            %out: Tensor = aten::add(%b_f, %mm, %1)
            return (%out))IR";

  // First find and replace aten::linear nodes with non-tensor bias values.
  replaceLinearWithBiasNonePattern(graph);

  torch::jit::SubgraphRewriter flatten_linear_to_linear;
  flatten_linear_to_linear.RegisterRewritePattern(flatten_linear_pattern, fused_linear);
  flatten_linear_to_linear.runOnGraph(graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
