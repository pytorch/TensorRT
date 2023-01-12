#include "torch/csrc/jit/passes/remove_dropout.h"
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph) {
  torch::jit::removeDropout(graph);
  LOG_GRAPH("Post remove dropout: " << *graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
