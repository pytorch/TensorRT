#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void UnpackAddMM(std::shared_ptr<torch::jit::Graph>& graph) {
  // TensorRT implicitly adds a flatten layer infront of FC layers if necessary
  std::string addmm_pattern = R"IR(
    graph(%b, %x, %w, %1):
      %out: Tensor = aten::addmm(%b, %x, %w, %1, %1)
      return (%out))IR";
  std::string mm_add_pattern = R"IR(
    graph(%b, %x, %w, %1):
      %mm: Tensor = aten::matmul(%x, %w)
      %bias: Tensor = trt::const(%b)
      %out: Tensor = aten::add(%bias, %mm, %1)
      return (%out))IR";

  torch::jit::SubgraphRewriter unpack_addmm;
  unpack_addmm.RegisterRewritePattern(addmm_pattern, mm_add_pattern);
  unpack_addmm.runOnGraph(graph);
  LOG_GRAPH("Post unpack addmm: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
