#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackHardSwish(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string hardswish_pattern = R"IR(
        graph(%input):
            %result = aten::hardswish(%input)
            return (%result))IR";

  std::string hardswish_pattern_inplace = R"IR(
        graph(%input):
            %result = aten::hardswish_(%input)
            return (%result))IR";

  std::string new_pattern = R"IR(
        graph(%input):
            %1 : Scalar = prim::Constant[value=3.]()
            %2 : Scalar = prim::Constant[value=1.]()
            %3 = aten::add(%input, %1, %2)
            %4 : Scalar = prim::Constant[value=0.]()
            %5 : Scalar = prim::Constant[value=6.]()
            %6 = aten::hardtanh(%3, %4, %5)
            %7 = aten::div(%6, %5)
            %8 = aten::mul(%input, %7)
            return (%8))IR";

  torch::jit::SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(hardswish_pattern, new_pattern);
  rewriter.RegisterRewritePattern(hardswish_pattern_inplace, new_pattern);
  rewriter.runOnGraph(graph);

  LOG_GRAPH("Post unpack hardswish: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
