#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

#include "core/util/prelude.h"

#include <vector>

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {
namespace {
using namespace torch::jit;
struct NOPRemoval {
  NOPRemoval(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    removeNode(graph_->block(), "aten::to");
    removeNode(graph_->block(), "aten::detach");
    torch::jit::EliminateDeadCode(graph_);
    LOG_DEBUG(
        "RemoveNOPs - Note: Removing remaining aten::to operators (in addition to other ops that have no meaning in TRT), if type casts need to be preserved, add a pass before this pass is run");
    LOG_GRAPH("Post aten::to removal: " << *graph_);
  }

 private:
  void removeNode(Block* b, std::string op) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == c10::Symbol::fromQualString(op)) {
        LOG_GRAPH("Found that node " << *n << "  is an " << op << " node (RemoveNOPs)" << std::endl);
        n->outputs()[0]->replaceAllUsesWith(n->inputs()[0]);
        it.destroyCurrent();
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};
} // namespace

void RemoveNOPs(std::shared_ptr<Graph> graph) {
  NOPRemoval tr(std::move(graph));
  tr.run();
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch