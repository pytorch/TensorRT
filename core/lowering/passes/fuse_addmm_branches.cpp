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
struct AddMMBranchFusion {
  AddMMBranchFusion(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    findAddMMVariantsNodes(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
    LOG_GRAPH("Post aten::addmm branch fusion: " << *graph_);
  }

 private:
  bool isAddMMVariantsNode(Node* n) {
    /// Check if this Node hosts a pattern like so:
    /// %ret : Tensor = prim::If(%622)
    /// block0():
    ///   %ret.1 : Tensor = aten::addmm(%self.fc.bias, %x9.1, %3677, %3, %3)
    ///   -> (%ret.1)
    /// block1():
    ///   %output.1 : Tensor = aten::matmul(%x9.1, %3677)
    ///   %output0.1 : Tensor = aten::add_(%output.1, %self.fc.bias, %3)
    ///   -> (%output0.1)

    if (n->blocks().size() != 2) {
      return false;
    }
    auto arm1 = n->blocks()[0];
    auto arm2 = n->blocks()[1];

    auto arm1_start = arm1->nodes().begin();
    auto arm2_start = arm2->nodes().begin();

    if ((*arm1_start)->kind().toQualString() == std::string("aten::addmm") &&
        (*(++arm1_start))->kind() == prim::Return &&
        (*arm2_start)->kind().toQualString() == std::string("aten::matmul") &&
        (*(++arm2_start))->kind().toQualString() != std::string("aten::add_") &&
        (*(++arm2_start))->kind() == prim::Return) {
      // Make sure that block0 is solely just the aten::addmm op and block1 is matmul + add
      return true;
    }

    return false;
  }

  void findAddMMVariantsNodes(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      if (n->kind() == prim::If && isAddMMVariantsNode(n)) {
        LOG_GRAPH("Found that node " << *n << " is an AddMM variants node (FuseAddMMBranches)" << std::endl);
        auto arm1 = n->blocks()[0];
        auto arm1_start = arm1->nodes().begin();

        auto input_values = (*arm1_start)->inputs();

        auto new_addmm_node = b->owningGraph()->create(c10::Symbol::fromQualString("aten::addmm"), input_values, 1);
        n->replaceAllUsesWith(new_addmm_node);

        auto old_insert_point = b->owningGraph()->insertPoint();
        b->owningGraph()->setInsertPoint(n);
        b->owningGraph()->insertNode(new_addmm_node);
        b->owningGraph()->setInsertPoint(old_insert_point);

        it.destroyCurrent();
      }
    }
  }

  std::shared_ptr<Graph> graph_;
};
} // namespace

void FuseAddMMBranches(std::shared_ptr<Graph> graph) {
  AddMMBranchFusion ammbf(std::move(graph));
  ammbf.run();
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
