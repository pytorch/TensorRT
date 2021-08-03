#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/create_functional_graphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/freeze_module.h"
#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/guard_elimination.h"
#include "torch/csrc/jit/passes/loop_unrolling.h"
#include "torch/csrc/jit/passes/lower_graph.h"
#include "torch/csrc/jit/passes/lower_tuples.h"
#include "torch/csrc/jit/passes/peephole.h"
#include "torch/csrc/jit/passes/remove_mutation.h"

#include "core/lowering/lowering.h"
#include "core/lowering/passes/passes.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace lowering {

void DropUnusedNodes(torch::jit::Block* b);

void LowerBlock(torch::jit::Block* b) {
  DropUnusedNodes(b);
}

void LowerGraph(std::shared_ptr<torch::jit::Graph>& g) {
  passes::UnpackHardSwish(g);
  torch::jit::EliminateRedundantGuards(g);
  torch::jit::RemoveListMutation(g);
  torch::jit::RemoveTensorMutation(g);
  torch::jit::CreateFunctionalGraphs(g);
  torch::jit::InlineFunctionalGraphs(g);
  torch::jit::PeepholeOptimize(g, false);
  passes::EliminateExceptionOrPassPattern(g);
  torch::jit::FuseLinear(g);
  torch::jit::LowerAllTuples(g);
  passes::RemoveContiguous(g);
  passes::RemoveDropout(g);
  passes::LinearToAddMM(g);
  passes::Conv2DToConvolution(g);
  passes::Conv3DToConvolution(g);
  passes::FuseAddMMBranches(g);
  passes::RemoveBNDimCheck(g);
  torch::jit::EliminateCommonSubexpression(g);
  // torch::jit::UnrollLoops(g);
  torch::jit::EliminateCommonSubexpression(g);
  passes::UnpackAddMM(g);
  // passes::UnpackBatchNorm(g);
  passes::UnpackLogSoftmax(g);
  passes::RemoveNOPs(g);
  passes::AliasOperators(g);
  passes::SiluToSigmoidMultipication(g);
  torch::jit::EliminateDeadCode(g);
  LOG_GRAPH(*g);
}

torch::jit::Module LowerModule(const torch::jit::Module& mod) {
  auto mod_ = torch::jit::freeze_module(mod);
  return mod_;
}

void NotateModuleForFallback(const torch::jit::Module& mod, std::string method_name, std::unordered_set<std::string> forced_fallback_modules) {
  auto named_submods = mod.modules();
  int mod_count = 0;
  for (const auto named_submod : named_submods) {
    auto mod_name = named_submod.type()->name()->qualifiedName().substr(10);
    std::size_t mangle_pos = mod_name.find("___torch_mangle_");
    if (mangle_pos != std::string::npos) {
      mod_name.erase(mangle_pos, 21);
    }
    if (mod_count == 0 && forced_fallback_modules.find(mod_name) != forced_fallback_modules.end()) {
      LOG_DEBUG("Marking module for fallback: " << mod_name);
      auto g = named_submod.get_method(method_name).graph();
      LOG_DEBUG(*g);
      auto nodes = g->block()->nodes();
      for (const auto n : nodes) {
        n->i_(c10::Symbol::attr("to_compile"), (int64_t) false);
      }
    } else if (mod_count > 0) {
      NotateModuleForFallback(named_submod, method_name, forced_fallback_modules);
    }
    mod_count++;
  }
}

std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>> Lower(
    const torch::jit::Module& mod,
    std::string method_name, const LowerInfo& lower_info) {
    LOG_DEBUG(lower_info);
  std::unordered_set<std::string> forced_fallback_modules(
      lower_info.forced_fallback_modules.begin(), lower_info.forced_fallback_modules.end());
  NotateModuleForFallback(mod, method_name, forced_fallback_modules);
  auto g = mod.get_method(method_name).graph();
  LOG_DEBUG("ARVIND before freeze: " << *g);
  auto lowered_mod = LowerModule(mod);
  g = lowered_mod.get_method(method_name).graph();
  LOG_DEBUG("ARVIND after freeze: " << *g);

  // Go through TRTorch Lowering to reformat graph to be conversion friendly
  // and also segment for accelerators and executors (TRT-DLA, TRT-GPU, PYT)
  LOG_GRAPH("TRTorch Graph Lowering");
  lowering::LowerGraph(g);
  //=[torch::jit::FoldConvBatchNorm2d(lowered_mod);
  LOG_GRAPH("LibTorch Lowering");
  auto graph_and_ivalues = torch::jit::LowerGraph(*g, lowered_mod._ivalue());
  // Is this necessary?
  lowering::LowerBlock(g->block());

  return graph_and_ivalues;
}

} // namespace lowering
} // namespace core
} // namespace trtorch
