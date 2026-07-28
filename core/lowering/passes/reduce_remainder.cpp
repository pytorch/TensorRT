#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void ReduceRemainder(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string remainder_pattern = R"IR(
        graph(%self : Tensor, %other : Tensor):
            %out : Tensor = aten::remainder(%self, %other)
            return (%out))IR";

  std::string remainder_reduce_pattern = R"IR(
        graph(%self : Tensor, %other : Tensor):
            %alpha : int = prim::Constant[value=1]()
            %floor: Tensor = aten::floor_divide(%self, %other)
            %prod: Tensor = aten::mul(%floor, %other)
            %out: Tensor = aten::sub(%self, %prod, %alpha)
            return (%out))IR";

  std::string remainder_scalar_pattern = R"IR(
        graph(%self : Tensor, %other : Scalar):
            %out : Tensor = aten::remainder(%self, %other)
            return (%out))IR";

  std::string remainder_scalar_reduce_pattern = R"IR(
        graph(%self : Tensor, %other : Scalar):
            %alpha : int = prim::Constant[value=1]()
            %floor: Tensor = aten::floor_divide(%self, %other)
            %prod: Tensor = aten::mul(%floor, %other)
            %out: Tensor = aten::sub(%self, %prod, %alpha)
            return (%out))IR";

  // replace aten::remainder with pointwise operations
  torch::jit::SubgraphRewriter map_remainder_to_pointwise_ops;
  map_remainder_to_pointwise_ops.RegisterRewritePattern(remainder_pattern, remainder_reduce_pattern);
  map_remainder_to_pointwise_ops.RegisterRewritePattern(remainder_scalar_pattern, remainder_scalar_reduce_pattern);
  map_remainder_to_pointwise_ops.runOnGraph(graph);

  LOG_GRAPH("Post lowering of [aten::remainder] -> " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
