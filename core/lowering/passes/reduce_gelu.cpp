#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void ReduceGelu(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string gelu_pattern = R"IR(
        graph(%x):
            %out : Tensor = aten::gelu(%x)
            return (%out))IR";

  std::string gelu_reduce_pattern = R"IR(
    graph(%x.1 : Tensor):
        %6 : float = prim::Constant[value=0.044714999999999998]()
        %5 : float = prim::Constant[value=0.79788456080000003]()
        %4 : float = prim::Constant[value=1.]()
        %3 : float = prim::Constant[value=0.5]()
        %2 : int = prim::Constant[value=1]()
        %7 : Tensor = aten::mul(%x.1, %3)
        %8 : Tensor = aten::mul(%x.1, %5)
        %9 : Tensor = aten::mul(%x.1, %6)
        %10 : Tensor = aten::mul(%9, %x.1)
        %11 : Tensor = aten::add(%10, %4, %2)
        %12 : Tensor = aten::mul(%8, %11)
        %13 : Tensor = aten::tanh(%12)
        %14 : Tensor = aten::add(%13, %4, %2)
        %15 : Tensor = aten::mul(%7, %14)
        return (%15))IR";

  // replace aten::gelu with pointwise operations
  torch::jit::SubgraphRewriter map_gelu_to_pointwise_ops;
  map_gelu_to_pointwise_ops.RegisterRewritePattern(gelu_pattern, gelu_reduce_pattern);
  map_gelu_to_pointwise_ops.runOnGraph(graph);

  LOG_GRAPH("Post lowering of [aten::gelu] -> " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
