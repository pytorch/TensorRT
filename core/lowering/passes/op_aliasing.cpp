#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void MapPadToConstantPadND(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string pad_pattern = R"IR(
        graph(%input, %pad, %value):
            %mode : str = prim::Constant[value="constant"]()
            %o : Tensor = aten::pad(%input, %pad, %mode, %value)
            return (%o))IR";
  std::string constant_pad_nd_pattern = R"IR(
        graph(%input, %pad, %value):
            %o : Tensor = aten::constant_pad_nd(%input, %pad, %value)
            return (%o))IR";
  torch::jit::SubgraphRewriter rewrite_pad;
  rewrite_pad.RegisterRewritePattern(pad_pattern, constant_pad_nd_pattern);
  rewrite_pad.runOnGraph(graph);

  // aten::pad can take None for value which is not support by constant_pad_nd
  // Legalize None to 0
  torch::jit::Value* const_zero = nullptr;
  for (auto n : graph->nodes()) {
    if (n->kind() == torch::jit::aten::constant_pad_nd) {
      if (n->inputs().size() < 3) {
        continue;
      }
      auto pad_value = n->input(2);
      if (pad_value->type()->isSubtypeOf(c10::NoneType::get())) {
        if (const_zero == nullptr) {
          const_zero = graph->insertConstant(0);
        }
        n->replaceInput(2, const_zero);
      }
    }
  }

  LOG_GRAPH("Post map pad(mode='constant') -> constant_pad_nd: " << *graph);
}

void AliasOperators(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string true_divide_pattern = R"IR(
        graph(%s, %o):
            %1 : Tensor = aten::true_divide(%s, %o)
            return (%1))IR";
  std::string div_pattern = R"IR(
        graph(%s, %o):
            %1 : Tensor = aten::div(%s, %o)
            return (%1))IR";

  torch::jit::SubgraphRewriter true_divide_to_div;
  true_divide_to_div.RegisterRewritePattern(true_divide_pattern, div_pattern);
  true_divide_to_div.runOnGraph(graph);
  LOG_GRAPH("Post map true_divide -> div: " << *graph);

  std::string scatter_sub_pattern = R"IR(
        graph(%data, %dim, %index, %value):
            %o : Tensor = aten::scatter_(%data, %dim, %index, %value)
            return (%o))IR";
  std::string scatter_pattern = R"IR(
        graph(%data, %dim, %index, %value):
            %o : Tensor = aten::scatter(%data, %dim, %index, %value)
            return (%o))IR";

  torch::jit::SubgraphRewriter rewrite_scatter;
  rewrite_scatter.RegisterRewritePattern(scatter_sub_pattern, scatter_pattern);
  rewrite_scatter.runOnGraph(graph);
  LOG_GRAPH("Post map scatter_ -> scatter: " << *graph);

  std::string multiply_pattern = R"IR(
        graph(%self, %other):
            %o : Tensor = aten::multiply(%self, %other)
            return (%o))IR";
  std::string mul_pattern = R"IR(
        graph(%self, %other):
            %o : Tensor = aten::mul(%self, %other)
            return (%o))IR";

  torch::jit::SubgraphRewriter rewrite_multiply;
  rewrite_multiply.RegisterRewritePattern(multiply_pattern, mul_pattern);
  rewrite_multiply.runOnGraph(graph);
  LOG_GRAPH("Post map multiply -> mul: " << *graph);

  MapPadToConstantPadND(graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
