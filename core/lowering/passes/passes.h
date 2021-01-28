#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace lowering {
namespace passes {

void Conv2DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void Conv3DToConvolution(std::shared_ptr<torch::jit::Graph>& graph);
void FuseAddMMBranches(std::shared_ptr<torch::jit::Graph> graph);
void FuseFlattenLinear(std::shared_ptr<torch::jit::Graph>& graph);
void EliminateExceptionOrPassPattern(std::shared_ptr<torch::jit::Graph> graph);
void RemoveBNDimCheck(std::shared_ptr<torch::jit::Graph> graph);
void RemoveContiguous(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveDropout(std::shared_ptr<torch::jit::Graph>& graph);
void RemoveNOPs(std::shared_ptr<torch::jit::Graph> graph);
void UnpackAddMM(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackBatchNorm(std::shared_ptr<torch::jit::Graph>& graph);
void UnpackLogSoftmax(std::shared_ptr<torch::jit::Graph>& graph);
void ElementWisePass(std::shared_ptr<torch::jit::Graph>& graph);

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace trtorch
