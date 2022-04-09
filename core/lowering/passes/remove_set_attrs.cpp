#include <stack>
#include <unordered_set>

#include "torch/csrc/jit/passes/subgraph_rewrite.h"

#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void RemoveSetAttrs(const torch::jit::Module& mod, std::string method_name) {
  auto g = mod.get_method(method_name).graph();

  std::string set_attr_pattern = R"IR(
        graph(%self, %0):
            None = prim::SetAttr[name="_has_warned"](%self, %0)
            return ())IR";
  std::string no_set_attr_pattern = R"IR(
        graph(%self, %0):
            return ())IR";

  // remove contiguous
  torch::jit::SubgraphRewriter remove_set_attr;
  remove_set_attr.RegisterRewritePattern(set_attr_pattern, no_set_attr_pattern);
  remove_set_attr.runOnGraph(g);
  LOG_GRAPH("Post remove contiguous: " << *g);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
