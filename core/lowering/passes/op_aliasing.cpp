#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace trtorch {
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
  ;

  // TODO
  // complete other element wise pass

  torch::jit::SubgraphRewriter true_divide_to_div;
  true_divide_to_div.RegisterRewritePattern(true_divide_pattern, div_pattern);
  true_divide_to_div.runOnGraph(graph);
  LOG_GRAPH("Post map true_divide -> div: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch