#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/dead_code_elimination.h"

#include "core/lowering/lowering.h"
#include "core/lowering/irfusers/irfusers.h"

namespace trtorch {
namespace core {
namespace lowering {

void DropUnusedNodes(torch::jit::Block* b);

void LowerBlock(torch::jit::Block* b) {
    DropUnusedNodes(b);
}

void LowerGraph(std::shared_ptr<torch::jit::Graph>& g) {
    torch::jit::FuseLinear(g);
    irfusers::RemoveDropout(g);
    irfusers::FuseFlattenLinear(g);
    irfusers::ExpandLogSoftmax(g);
    //irfusers::UnpackBatchNorm(g);
    //torch::jit::EliminateDeadCode(g);
}
    
} // namespace lowering
} // namespace core
} // namespace trtorch
