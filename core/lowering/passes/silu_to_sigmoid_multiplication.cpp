#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void SiluToSigmoidMultipication(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string silu_pattern = R"IR(
        graph(%x):
            %1 : Tensor = aten::silu(%x)
            return (%1))IR";
  std::string sigmoid_multiplication_pattern = R"IR(
        graph(%x):
            %1 : Tensor = aten::sigmoid(%x)
            %2 : Tensor = aten::mul(%x, %1)
            return (%2))IR";
  ;

  torch::jit::SubgraphRewriter map_silu_to_sigmoid_multiplication;
  map_silu_to_sigmoid_multiplication.RegisterRewritePattern(silu_pattern, sigmoid_multiplication_pattern);
  map_silu_to_sigmoid_multiplication.runOnGraph(graph);
  LOG_GRAPH("Post map silu -> x * sigmoid(x): " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
