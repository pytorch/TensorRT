#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators trt_const_op_reg({
        Operator(
            "trt::const(Tensor val) -> Tensor",
            [](Stack& stack) {
                return 0; //nop
            },
            aliasAnalysisFromSchema())});

} // namespace jit
} // namespace torch

namespace trtorch {
namespace core {
namespace lowering {
namespace irfusers {

// // May be abusing aten::_tensor_to_list(Tensor self) -> int[]
// // Treating it as an emit_constant by the converters
// // We could register a custom op (trt::emit_constant) which we can use to convert
// // constant tensors to TRT ITensors
void UnpackBatchNorm(std::shared_ptr<torch::jit::Graph>& graph) {
    // Convert BatchNorm into individual operators
    // batch_norm = gamma * (in - mu) / sqrt(var + epsilon) + beta
    std::string batch_norm_pattern = R"IR(
       graph(%input, %gamma, %beta, %mean,
             %var, %training, %momentum, %eps, %cudnn):
           %1 = aten::batch_norm(%input, %gamma, %beta, %mean, %var, %training, %momentum, %eps, %cudnn)
           return (%1))IR";

    std::string expanded_batch_norm_pattern = R"IR(
        graph(%input, %gamma, %beta, %mean,
              %var, %training, %momentum, %eps, %cudnn):
            %gamma_trt = trt::const(%gamma)
            %beta_trt = trt::const(%beta)
            %mean_trt = trt::const(%mean)
            %var_trt = trt::const(%var)
            %0: Scalar = prim::Constant[value=1]()
            %1 = aten::sub(%input, %mean_trt, %0)
            %2: Scalar = prim::Constant[value=1]()
            %3 = aten::add(%var_trt, %eps, %2)
            %4 = aten::sqrt(%3)
            %5 = aten::div(%1, %4)
            %6 = aten::mul(%gamma_trt, %5)
            %7: Scalar = prim::Constant[value=1]()
            %8 = aten::add(%6, %beta_trt, %7)
            return(%8))IR";

    torch::jit::SubgraphRewriter unpack_batch_norm;
    unpack_batch_norm.RegisterRewritePattern(batch_norm_pattern, expanded_batch_norm_pattern);
    unpack_batch_norm.runOnGraph(graph);
}
} // Namespace Irfusers
} // namespace lowering
} // namespace core
} // namespace trtorch
