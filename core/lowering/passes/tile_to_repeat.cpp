#include "core/util/prelude.h"
#include "torch/csrc/jit/passes/subgraph_rewrite.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {
void ReplaceTileWithRepeat(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string tile_pattern = R"IR(
                        graph(%input, %1):
                            %2 = aten::tile(%input, %1)
                            return (%2))IR";
  std::string repeat_pattern = R"IR(
                        graph(%input, %1):
                            %2 = aten::repeat(%input, %1)
                            return (%2))IR";
  torch::jit::SubgraphRewriter tile_to_repeat;
  tile_to_repeat.RegisterRewritePattern(tile_pattern, repeat_pattern);
  tile_to_repeat.runOnGraph(graph);
  LOG_GRAPH("Mapping tile -> repeat: " << *graph);
}
} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
