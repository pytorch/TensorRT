#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void RemoveInplaceAdd(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string inplace_add_pattern = R"IR(
        graph(%self, %other, %1):
            %out = aten::add_(%self, %other, %1)
            return (%out))IR";
  std::string normal_add_pattern = R"IR(
        graph(%self, %other, %1):
            %out = aten::add(%self, %other, %1)
            return (%out))IR";

  torch::jit::SubgraphRewriter remove_inplace_add;
  remove_inplace_add.RegisterRewritePattern(inplace_add_pattern, normal_add_pattern);
  remove_inplace_add.runOnGraph(graph);

  LOG_GRAPH("Post remove inplace add: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
