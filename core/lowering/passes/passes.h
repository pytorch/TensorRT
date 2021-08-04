#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void NotateModuleForFallback(const torch::jit::Module& mod, std::string mod_name, const std::string& method_name, std::unordered_set<std::string> forced_fallback_modules);

void Conv2DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void Conv3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void FuseAddMMBranches(std::shared_ptr<torch::jit::Graph> graph);
void LinearToAddMM(std::shared_ptr<torch::jit::Graph>& graph);
void EliminateExceptionOrPassPattern(std::shared_ptr<torch::jit::Graph> graph);
void MarkNodesForFallback(std::shared_ptr<torch::jit::Graph>& g);
void RemoveBNDimCheck(std::shared_ptr<torch::jit::Graph> graph);
void RemoveContiguous(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveNOPs(std::shared_ptr<torch::jit::Graph> graph);
void UnpackAddMM(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackBatchNorm(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackLogSoftmax(std::shared_ptr<torch::jit::Graph>& graph);
void AliasOperators(std::shared_ptr<torch::jit::Graph>& graph);
void SiluToSigmoidMultipication(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackHardSwish(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
