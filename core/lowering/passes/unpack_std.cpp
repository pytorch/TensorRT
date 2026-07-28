#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackStd(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string std_pattern = R"IR(
    graph(%1, %dim, %unbiased, %keepdim):
      %out: Tensor = aten::std(%1, %dim, %unbiased, %keepdim)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%1, %dim, %unbiased, %keepdim):
      %z: Tensor = aten::var(%1, %dim, %unbiased, %keepdim)
      %out: Tensor = aten::sqrt(%z)
      return (%out))IR";

  torch::jit::SubgraphRewriter std_rewriter;
  std_rewriter.RegisterRewritePattern(std_pattern, unpacked_pattern);
  std_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack std: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
