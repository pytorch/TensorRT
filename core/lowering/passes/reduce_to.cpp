#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void ReduceToOperation(std::shared_ptr<torch::jit::Graph>& graph) {
  std::string to_device_pattern = R"IR(
        graph(%x, %device, %dtype, %nb, %copy, %format):
            %out : Tensor = aten::to(%x, %device, %dtype, %nb, %copy, %format)
            return (%out))IR";
  std::string to_general_pattern = R"IR(
        graph(%x, %device, %dtype, %nb, %copy, %format):
            %out : Tensor = aten::to(%x, %dtype, %nb, %copy, %format)
            return (%out))IR";

  // replace aten::to.device with aten::to.dtype
  torch::jit::SubgraphRewriter map_aten_device_to_dtype;
  map_aten_device_to_dtype.RegisterRewritePattern(to_device_pattern, to_general_pattern);
  map_aten_device_to_dtype.runOnGraph(graph);
  LOG_GRAPH("Post lowering of aten::to.device -> " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
