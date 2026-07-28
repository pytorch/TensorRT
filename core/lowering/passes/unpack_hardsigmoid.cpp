#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackHardSigmoid(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string hardsigmoid_pattern = R"IR(
        graph(%input):
            %result = aten::hardsigmoid(%input)
            return (%result))IR";

  std::string hardsigmoid_pattern_inplace = R"IR(
        graph(%input):
            %result = aten::hardsigmoid_(%input)
            return (%result))IR";

  std::string new_pattern = R"IR(
        graph(%x.1):
            %22 : float = prim::Constant[value=0.5]()
            %3 : int = prim::Constant[value=6]()
            %5 : int = prim::Constant[value=1]()
            %10 : int = prim::Constant[value=0]()
            %4 : Tensor = aten::div(%x.1, %3)
            %9 : Tensor = aten::add(%4, %22, %5)
            %21 : Tensor = aten::clamp(%9, %10, %5)
            return (%21))IR";

  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(hardsigmoid_pattern, new_pattern);
  rewriter.RegisterRewritePattern(hardsigmoid_pattern_inplace, new_pattern);
  rewriter.runOnGraph(graph);

  LOG_GRAPH("Post unpack hardsigmoid: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
