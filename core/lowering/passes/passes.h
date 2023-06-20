#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void NotateModuleForFallback(
    const torch::jit::Module& mod,
    std::string mod_name,
    std::string method_name,
    std::unordered_set<std::string> forced_fallback_modules);
void Conv1DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void ConvTransposed1DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void Conv2DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void ConvTransposed2DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void Conv3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void ConvTransposed3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void FuseAddMMBranches(std::shared_ptr<torch::jit::Graph> graph);
void LinearToAddMM(std::shared_ptr<torch::jit::Graph>& graph);
void EliminateExceptionOrPassPattern(std::shared_ptr<torch::jit::Graph> graph);
void ReduceToOperation(std::shared_ptr<torch::jit::Graph>& graph);
void ReduceGelu(std::shared_ptr<torch::jit::Graph>& graph);
void ReduceRemainder(std::shared_ptr<torch::jit::Graph>& graph);
void MarkNodesForFallback(std::shared_ptr<torch::jit::Graph>& g, bool delete_delims);
void RemoveBNDimCheck(std::shared_ptr<torch::jit::Graph> graph);
void RemoveContiguous(std::shared_ptr<torch::jit::Graph>& graph);
void ViewToReshape(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveNOPs(std::shared_ptr<torch::jit::Graph> graph);
void RemoveSingleUse0DTensors(std::shared_ptr<torch::jit::Graph>& g);
void RemoveUnnecessaryCasts(std::shared_ptr<torch::jit::Graph>& graph);
void ReplaceAtenInt(std::shared_ptr<torch::jit::Graph>& g);
void RemoveCollectionCast(std::shared_ptr<torch::jit::Graph>& g);
void UnpackAddMM(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackBatchNorm(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackLogSoftmax(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackRsqrt(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackStd(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackVar(std::shared_ptr<torch::jit::Graph>& graph);
void AliasOperators(std::shared_ptr<torch::jit::Graph>& graph);
void SiluToSigmoidMultipication(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackHardSwish(std::shared_ptr<torch::jit::Graph>& graph);
void RewriteInputsWithParams(std::shared_ptr<torch::jit::Graph>& g, std::vector<torch::jit::IValue>& params);
void UnpackHardSigmoid(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackAndCastMaskedFill(std::shared_ptr<torch::jit::Graph>& graph, std::string target_device_name);
void UnpackAndCastNumToTensor(std::shared_ptr<torch::jit::Graph>& graph, std::string target_device_name);
void UnpackAndCastFull(std::shared_ptr<torch::jit::Graph>& graph, std::string target_device_name);
void ReplaceScalarImplicit(std::shared_ptr<torch::jit::Graph>& graph);
void ReplaceAtenPad(std::shared_ptr<torch::jit::Graph>& graph);

// utility functions exposed for testing
std::string unmangle_cls_name(const std::string& name);

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
