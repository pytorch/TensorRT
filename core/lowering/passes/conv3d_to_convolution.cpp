#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void Conv3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string conv3d_pattern = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %4 : Tensor = aten::conv3d(%x, %w, %b, %s, %p, %d, %g)
            return (%4))IR";
  std::string convolution_pattern = R"IR(
        graph(%x, %w, %b, %s, %p, %d, %g):
            %1 : bool = prim::Constant[value=0]()
            %2 : int[] = prim::Constant[value=[0, 0, 0]]()
            %4 : Tensor = aten::_convolution(%x, %w, %b, %s, %p, %d, %1, %2, %g, %1, %1, %1, %1)
            return (%4))IR";
  ;

  // replace matmul + add pattern to linear
  torch::jit::SubgraphRewriter map_conv3d_to_convolution;
  map_conv3d_to_convolution.RegisterRewritePattern(conv3d_pattern, convolution_pattern);
  map_conv3d_to_convolution.runOnGraph(graph);
  LOG_GRAPH("Post map conv3d -> _convolution: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch