#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void FuseFlattenLinear(std::shared_ptr<torch::jit::Graph>& graph) {
    //TensorRT implicitly adds a flatten layer infront of FC layers if necessary
    std::string flatten_linear_pattern = R"IR(
        graph(%input, %6, %7, %weight, %bias):
            %flat = aten::flatten(%input, %6, %7)
            %res = aten::linear(%flat, %weight, %bias)
            return (%res))IR";
    std::string flatten_linear_bias_none_pattern = R"IR(
        graph(%input, %6, %7, %weight):
            %flat = aten::flatten(%input, %6, %7)
            %bias: Tensor? = prim::Constant()
            %res = aten::linear(%flat, %weight, %bias)
            return (%res))IR";
    std::string fused_linear = R"IR(
        graph(%input, %6, %7, %weight, %bias):
            %res = aten::linear(%input, %weight, %bias)
            return (%res))IR";

    std::string fused_linear_bias_none = R"IR(
        graph(%input, %6, %7, %weight):
            %bias: Tensor? = prim::Constant()
            %res = aten::linear(%input, %weight, %bias)
            return (%res))IR";

    torch::jit::SubgraphRewriter flatten_linear_to_linear;
    flatten_linear_to_linear.RegisterRewritePattern(flatten_linear_pattern, fused_linear);
    flatten_linear_to_linear.runOnGraph(graph);

    torch::jit::SubgraphRewriter flatten_linear_bias_none_to_linear;
    flatten_linear_bias_none_to_linear.RegisterRewritePattern(
        flatten_linear_bias_none_pattern, fused_linear_bias_none);
    flatten_linear_bias_none_to_linear.runOnGraph(graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
