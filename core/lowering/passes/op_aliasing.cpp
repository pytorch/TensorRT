#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

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
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
