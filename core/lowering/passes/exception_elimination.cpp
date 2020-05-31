#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/runtime/graph_executor.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

#include "core/util/prelude.h"

#include <vector>

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {
namespace {
using namespace torch::jit;
struct ExceptionOrPassPatternElimination {
  ExceptionOrPassPatternElimination(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {}

  void run() {
    findExceptionOrPassNodes(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
    LOG_GRAPH("Post exeception or pass elimination: " << *graph_);
  }

private:
  bool isExceptionOrPassNode(Node* n) {
    /// Check if this Node hosts a pattern like so:
    ///  = prim::If(%5958)
    ///   block0():
    ///     = prim::RaiseException(%45)
    ///    -> ()
    ///   block1():
    ///    -> ()
    if (n->blocks().size() != 2) {
      return false;
    }
    auto arm1 = n->blocks()[0];
    auto arm2 = n->blocks()[1];
    if (arm1->outputs().size() != 0 || arm2->outputs().size() != 0) {
      // Make sure that the node doesn't actually produce any Value that are used by other nodes
      return false;
    }

    auto arm1_start = arm1->nodes().begin();

    if ((*arm1_start)->kind() != prim::RaiseException && (*(++arm1_start))->kind() != prim::Return) {
      // Make sure that block0 is solely just the exception and the return
      return false;
    }

    if ((*(arm2->nodes().begin()))->kind() != prim::Return) {
      // Make sure that block1 is solely the return
      return false;
    }

    return true;
  }

  void findExceptionOrPassNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::If && isExceptionOrPassNode(n)) {
        LOG_GRAPH("Found that node " << *n << "  is an exception or pass node (EliminateChecks)" << std::endl);
        it.destroyCurrent();
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};
} // namespace

void EliminateExceptionOrPassPattern(std::shared_ptr<Graph> graph) {
  ExceptionOrPassPatternElimination eppe(std::move(graph));
  eppe.run();
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
