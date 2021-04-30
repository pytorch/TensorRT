#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void LinearToAddMM(std::shared_ptr<torch::jit::Graph>& graph) {
  // TensorRT implicitly adds a flatten layer infront of FC layers if necessary
  std::string flatten_linear_pattern = R"IR(
        graph(%input, %weight, %bias):
            %res = aten::linear(%input, %weight, %bias)
            return (%res))IR";
  std::string flatten_linear_bias_none_pattern = R"IR(
        graph(%input, %weight):
            %bias: Tensor? = prim::Constant()
            %res = aten::linear(%input, %weight, %bias)
            return (%res))IR";

  std::string fused_linear = R"IR(
        graph(%input, %weight_t, %bias):
            %1: int = prim::Constant[value=1]()
            %weight = aten::t(%weight_t)
            %mm: Tensor = aten::matmul(%input, %weight)
            %b_f: Tensor = trt::const(%bias)
            %out: Tensor = aten::add_(%b_f, %mm, %1)
            return (%out))IR";
  std::string fused_linear_bias_none = R"IR(
        graph(%input, %weight_t):
            %weight = aten::t(%weight_t)
            %mm: Tensor = aten::matmul(%input, %weight)
            return (%mm))IR";

  torch::jit::SubgraphRewriter flatten_linear_to_linear;
  flatten_linear_to_linear.RegisterRewritePattern(flatten_linear_pattern, fused_linear);
  flatten_linear_to_linear.runOnGraph(graph);

  torch::jit::SubgraphRewriter flatten_linear_bias_none_to_linear;
  flatten_linear_bias_none_to_linear.RegisterRewritePattern(flatten_linear_bias_none_pattern, fused_linear_bias_none);
  flatten_linear_bias_none_to_linear.runOnGraph(graph);
  LOG_GRAPH("Post linear to addmm: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
