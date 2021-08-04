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
  passes::MarkNodesForFallback(g);
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

torch::jit::Module LowerModule(const torch::jit::Module& mod, const std::string& method_name, std::unordered_set<std::string> forced_fallback_modules) {
  passes::NotateModuleForFallback(mod, "", method_name, forced_fallback_modules);
  LOG_GRAPH("Post notation pass: " << *mod.get_method(method_name).graph());
  auto mod_ = torch::jit::freeze_module(mod);
  LOG_GRAPH("Post freeze: " << *mod_.get_method(method_name).graph());
  return mod_;
}

std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>> Lower(
    const torch::jit::Module& mod,
    std::string method_name, const LowerInfo& lower_info) {
    LOG_DEBUG(lower_info);
    LOG_GRAPH(*mod.get_method(method_name).graph());
  std::unordered_set<std::string> forced_fallback_modules(
      lower_info.forced_fallback_modules.begin(), lower_info.forced_fallback_modules.end());
  auto lowered_mod = LowerModule(mod, method_name, forced_fallback_modules);
  auto g = lowered_mod.get_method(method_name).graph();
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
