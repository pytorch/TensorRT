#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void ViewToReshape(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string view_pattern = R"IR(
        graph(%x, %1):
            %out : Tensor = aten::view(%x, %1)
            return (%out))IR";

  std::string reshape_pattern = R"IR(
        graph(%x, %1):
            %out : Tensor = aten::reshape(%x, %1)
            return (%out))IR";

  // replace aten::view with aten::reshape
  torch::jit::SubgraphRewriter map_view_to_reshape;
  map_view_to_reshape.RegisterRewritePattern(view_pattern, reshape_pattern);
  map_view_to_reshape.runOnGraph(graph);

  LOG_GRAPH("Post lowering of aten::view -> " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
