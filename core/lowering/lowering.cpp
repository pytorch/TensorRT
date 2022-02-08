#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/create_functional_graphs.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/erase_number_types.h"
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

namespace torch_tensorrt {
namespace core {
namespace lowering {

void DropUnusedNodes(torch::jit::Block* b);

void LowerBlock(torch::jit::Block* b) {
  DropUnusedNodes(b);
}

void LowerGraph(std::shared_ptr<torch::jit::Graph>& g, LowerInfo lower_info) {
  torch::jit::EliminateRedundantGuards(g);
  torch::jit::RemoveListMutation(g);
  torch::jit::RemoveTensorMutation(g);
  torch::jit::CreateFunctionalGraphs(g);
  torch::jit::InlineFunctionalGraphs(g);
  torch::jit::PeepholeOptimize(g, false);
  torch::jit::FuseLinear(g);
  torch::jit::LowerAllTuples(g);
  if (!lower_info.disable_cse) {
    torch::jit::EliminateCommonSubexpression(g);
  }
  torch::jit::EliminateDeadCode(g);
  if (lower_info.forced_fallback_modules.size() > 0) {
    passes::MarkNodesForFallback(g, true);
  }
  passes::UnpackHardSwish(g);
  passes::EliminateExceptionOrPassPattern(g);
  passes::ReduceToOperation(g);
  passes::ReduceGelu(g);
  passes::RemoveContiguous(g);
  passes::ViewToReshape(g);
  passes::RemoveDropout(g);
  passes::LinearToAddMM(g);
  passes::Conv1DToConvolution(g);
  passes::ConvTransposed1DToConvolution(g);
  passes::Conv2DToConvolution(g);
  passes::Conv3DToConvolution(g);
  passes::FuseAddMMBranches(g);
  passes::RemoveBNDimCheck(g);
  // torch::jit::UnrollLoops(g);
  passes::UnpackAddMM(g);
  // passes::UnpackBatchNorm(g);
  passes::UnpackLogSoftmax(g);
  passes::UnpackStd(g);
  passes::UnpackVar(g);
  passes::RemoveNOPs(g);
  passes::AliasOperators(g);
  passes::SiluToSigmoidMultipication(g);
  passes::RemoveUnnecessaryCasts(g);
  LOG_GRAPH(*g);
}

torch::jit::Module LowerModule(const torch::jit::Module& mod, std::string method_name, const LowerInfo& lower_info) {
  std::unordered_set<std::string> forced_fallback_modules(
      lower_info.forced_fallback_modules.begin(), lower_info.forced_fallback_modules.end());
  if (forced_fallback_modules.size() > 0) {
    passes::NotateModuleForFallback(mod, "", method_name, forced_fallback_modules);
    LOG_GRAPH("After MLF notation pass: " << *mod.get_method(method_name).graph());
  }
  auto mod_ = torch::jit::freeze_module(mod);
  LOG_GRAPH("After freeze: " << *mod_.get_method(method_name).graph());
  return mod_;
}

std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>> Lower(
    const torch::jit::Module& mod,
    std::string method_name,
    const LowerInfo& lower_info) {
  LOG_DEBUG(lower_info);
  LOG_GRAPH("Before lowering: " << *mod.get_method(method_name).graph());
  auto lowered_mod = lower_info.unfreeze_module ? mod : LowerModule(mod, method_name, lower_info);
  auto g = lowered_mod.get_method(method_name).graph();

  LOG_GRAPH("LibTorch Lowering");
  auto graph_and_ivalues = torch::jit::LowerGraph(*g, lowered_mod._ivalue());

  // Go through Torch-TensorRT Lowering to reformat graph to be conversion friendly
  // and also segment for accelerators and executors (TRT-DLA, TRT-GPU  , PYT)
  // unfreeze_module is used to not perform constant folding on weights in the network.
  // In quantization aware trained (QAT) models, weights are passed through quantize and
  // dequantize nodes which should not be folded. So unfreeze_module is set to True for QAT models.
  LOG_GRAPH("Torch-TensorRT.TorchScript Graph Lowering");
  lowering::LowerGraph(graph_and_ivalues.first, lower_info);

  // Is this necessary?
  // lowering::LowerBlock(g->block());

  LOG_INFO("Lowered Graph: " << *(graph_and_ivalues.first));
  return graph_and_ivalues;
}

} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
