#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

#include <vector>

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {


// Presumably this is safe since torch::jit::EraseNumberTypesOnBlock exists which just
// removes prim::TensorToNum, aten::Float, aten::Int and prim::NumToTensor nodes outright
void RemoveUnnecessaryCasts(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string int_cast_pattern = R"IR(
    graph(%1: int):
      %2: Tensor = aten::NumToTensor(%1)
      %3: int = aten::Int(%2)
      return (%3))IR";
  std::string int_clean_pattern = R"IR(
    graph(%1: int):
      return (%1))IR";

  std::string float_cast_pattern = R"IR(
    graph(%1: float):
      %2: Tensor = aten::NumToTensor(%1)
      %3: float = aten::Float(%2)
      return (%3))IR";
  std::string float_clean_pattern = R"IR(
    graph(%1: float):
      return (%1))IR";

  std::string bool_cast_pattern = R"IR(
    graph(%1: bool):
      %2: Tensor = aten::NumToTensor(%1)
      %3: bool = aten::Bool(%2)
      return (%3))IR";
  std::string bool_clean_pattern = R"IR(
    graph(%1: bool):
      return (%1))IR";

  torch::jit::SubgraphRewriter int_cast_rewriter;
  int_cast_rewriter.RegisterRewritePattern(int_cast_pattern, int_clean_pattern);
  int_cast_rewriter.runOnGraph(graph);

  torch::jit::SubgraphRewriter float_cast_rewriter;
  float_cast_rewriter.RegisterRewritePattern(float_cast_pattern, float_clean_pattern);
  float_cast_rewriter.runOnGraph(graph);

  torch::jit::SubgraphRewriter bool_cast_rewriter;
  bool_cast_rewriter.RegisterRewritePattern(bool_cast_pattern, bool_clean_pattern);
  bool_cast_rewriter.runOnGraph(graph);

  LOG_GRAPH("After RemoveUnnecessaryCasts: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
