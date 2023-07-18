#include "torch/csrc/jit/ir/alias_analysis.h"
#include "torch/csrc/jit/jit_log.h"
#include "torch/csrc/jit/passes/constant_pooling.h"
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
struct ExceptionOrPassPatternElimination {
  ExceptionOrPassPatternElimination(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {}

  void run() {
    findExceptionOrPassNodes(graph_->block());
    torch::jit::EliminateDeadCode(graph_);
    LOG_GRAPH("Post exeception or pass elimination: " << *graph_);
  }

 private:
  bool isExceptionOrPassNode(Node* n) {
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
    auto arm2_start = arm2->nodes().begin();

    bool arm1_starts_with_exception = (*arm1_start)->kind() == prim::RaiseException;
    bool arm2_starts_with_exception = (*arm2_start)->kind() == prim::RaiseException;

    // if (!arm1_starts_with_exception && !arm2_starts_with_exception) {
    // Neither arm matches the pattern
    //   return false;
    //}

    /// Check if this Node hosts a pattern like so:
    ///  = prim::If(%5958)
    ///   block0():
    ///     = prim::RaiseException(%45)
    ///    -> ()
    ///   block1():
    ///    -> ()
    if (arm1_starts_with_exception) {
      if ((*(++arm1_start))->kind() == prim::Return) {
        // Make sure that block0 is solely just the exception and the return
        if ((*(arm2_start))->kind() == prim::Return) {
          // Make sure that block1 is solely the return
          return true;
        }
      }
    }

    /// Check if this Node hosts a pattern like so:
    ///  = prim::If(%5958)
    ///   block0():
    ///    -> ()
    ///   block1():
    ///     = prim::RaiseException(%45)
    ///    -> ()
    if (arm2_starts_with_exception) {
      if ((*(++arm2_start))->kind() == prim::Return) {
        // Make sure that block1 is solely just the exception and the return
        if ((*(arm1_start))->kind() == prim::Return) {
          // Make sure that block0 is solely the return
          return true;
        }
      }
    }

    return false;
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
  if (graph) {
    LOG_GRAPH("Post Eliminate Exception or Pass Patterns: " << *graph);
  }
}

/*
  Below is a fork of the torch::jit::EliminateExceptions pass, with node replacement
  using replaceAllUsesDominatedByNodeWith instead of replaceAllUsesWith,
  so as to not invalidate the IR in challenging cases, such as nested Ifs

  Original Source from which it was adapted:
  https://github.com/pytorch/pytorch/blob/c29ab84115f40614d04e4557ea2e1ac40b7aa75c/torch/csrc/jit/passes/remove_exceptions.cpp
*/

bool certainlyThrows(Block* block) {
  // A block certainly throws an exception if it contains
  // the prim::RaiseException operation
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::RaiseException) {
      return true;
    }
  }
  return false;
}

void EliminateExceptionsSafe(Block* block) {
  auto graph = block->owningGraph();
  // Generate false and true constant placeholders
  Value* false_const = graph->insertConstant(IValue(false));
  Value* true_const = graph->insertConstant(IValue(true));

  // For each prim::If node, if either block certainly throws an exception,
  // replace input conditional of the node input with the logical opposite
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::If) {
      Block* true_block = n->blocks()[0];
      Block* false_block = n->blocks()[1];
      bool removed_exception = false;
      Value* input_value_replacement;

      // If the block throws an exception, replace input with logical opposite
      if (certainlyThrows(true_block)) {
        removed_exception = true;
        input_value_replacement = false_const;
      } else if (certainlyThrows(false_block)) {
        removed_exception = true;
        input_value_replacement = true_const;
      }

      // Log node and perform input replacement
      if (removed_exception) {
        LOG_WARNING("Detected and removing exception in TorchScript IR for node: " << util::node_info(n));
        n->insertInput(0, input_value_replacement);
        n->removeInput(1);
      }
    }

    // Inspect and replace all instances within subblocks of the current node
    for (Block* subblock : n->blocks()) {
      EliminateExceptionsSafe(subblock);
    }
  }
}

void EliminateExceptionsSafe(std::shared_ptr<Graph>& graph) {
  EliminateExceptionsSafe(graph->block());
  ConstantPropagation(graph);
  ConstantPooling(graph);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
