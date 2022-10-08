#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void UnpackVar(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string var_pattern = R"IR(
    graph(%input, %dim, %unbiased, %keepdim):
      %out: Tensor = aten::var(%input, %dim, %unbiased, %keepdim)
      return (%out))IR";
  std::string unpacked_pattern = R"IR(
    graph(%input, %dims, %unbiased, %keepdim):
      %none: None = prim::Constant()
      %false: bool = prim::Constant[value=0]()
      %0: int = prim::Constant[value=0]()
      %f32_dtype: int = prim::Constant[value=6]()
      %1: int = prim::Constant[value=1]()
      %sqrd: Tensor = aten::mul(%input, %input)
      %sqrdmean: Tensor = aten::mean(%sqrd, %dims, %keepdim, %none)
      %mean: Tensor = aten::mean(%input, %dims, %keepdim, %none)
      %meansqrd: Tensor = aten::mul(%mean, %mean)
      %var: Tensor = aten::sub(%sqrdmean, %meansqrd, %1)
      %varout : Tensor = prim::If(%unbiased)
        block0():
          %originalshape: int[] = aten::size(%input)
          %originalshapet: Tensor = aten::tensor(%originalshape, %f32_dtype, %none, %false)
          %originalnumel: Tensor = aten::prod(%originalshapet, %0, %false, %none)

          %resultingshape: int[] = aten::size(%var)
          %resultingshapet: Tensor = aten::tensor(%resultingshape, %f32_dtype, %none, %false)
          %resultingnumel: Tensor = aten::prod(%resultingshapet, %0, %false, %none)

          %numel: Tensor = aten::div(%originalnumel, %resultingnumel)

          %mul: Tensor = aten::mul(%var, %numel)
          %sub: Tensor = aten::sub(%numel, %1, %1)
          %v: Tensor = aten::div(%mul, %sub)
          -> (%v)
        block1():
          -> (%var)
      return(%varout))IR";

  torch::jit::SubgraphRewriter var_rewriter;
  var_rewriter.RegisterRewritePattern(var_pattern, unpacked_pattern);
  var_rewriter.runOnGraph(graph);
  LOG_GRAPH("Post unpack var: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
