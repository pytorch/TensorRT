#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackRsqrt(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string rsqrt_pattern = R"IR(
    graph(%1):
      %out: Tensor = aten::rsqrt(%1)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%1):
      %intermediate: Tensor = aten::sqrt(%1)
      %out: Tensor = aten::reciprocal(%intermediate)
      return (%out))IR";

  torch::jit::SubgraphRewriter rsqrt_rewriter;
  rsqrt_rewriter.RegisterRewritePattern(rsqrt_pattern, unpacked_pattern);
  rsqrt_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack rsqrt: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
