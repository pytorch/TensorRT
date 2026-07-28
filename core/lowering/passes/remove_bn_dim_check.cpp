#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/constant_propagation.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/runtime/graph_executor.h"

#include "core/util/prelude.h"

#include <vector>

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {
namespace {
using namespace torch::jit;
struct BNDimCheckRemoval {
  BNDimCheckRemoval(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    findBNDimCheckNodes(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
    LOG_GRAPH("Post batch norm dim check removal: " << *graph_);
  }

 private:
  bool isBNDimCheckNodes(Node* n) {
    /// Check if this Node hosts a pattern like so:
    /// %290 : bool = aten::ne(%289, %9)
    ///   = prim::If(%290)
    ///     block0():
    ///       %291 : str = aten::format(%10, %289)
    ///        = prim::RaiseException(%291)
    ///       -> ()
    ///     block1():
    ///       -> ()

    if (n->blocks().size() != 2) {
      return false;
    }
    auto arm1 = n->blocks()[0];
    auto arm2 = n->blocks()[1];
    if (arm1->outputs().size() != 0 || arm2->outputs().size() != 0) {
      // Make sure that the node doesn't actually produce any Value that are
      // used by other nodes
      return false;
    }

    auto arm1_start = arm1->nodes().begin();

    if ((*arm1_start)->kind() != c10::Symbol::fromQualString("aten::format") &&
        (*(++arm1_start))->kind() != prim::RaiseException && (*(++arm1_start))->kind() != prim::Return) {
      // Make sure that block0 is solely just the exception and the return
      return false;
    }

    if ((*(arm2->nodes().begin()))->kind() != prim::Return) {
      // Make sure that block1 is solely the return
      return false;
    }

    return true;
  }

  void findBNDimCheckNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::If && isBNDimCheckNodes(n)) {
        LOG_GRAPH("Found that node " << *n << "  is an batch norm dim check node (EliminateChecks)" << std::endl);
        it.destroyCurrent();
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};
} // namespace

void RemoveBNDimCheck(std::shared_ptr<Graph> graph) {
  BNDimCheckRemoval bndcr(std::move(graph));
  bndcr.run();
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
