#include "torch/csrc/jit/passes/dead_code_elimination.h"
#include "torch/csrc/jit/passes/fuse_linear.h"
#include "torch/csrc/jit/passes/lower_graph.h"
#include "torch/csrc/jit/passes/quantization.h"

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

void LowerModule(const torch::jit::script::Module& mod) {
    torch::jit::FoldConvBatchNorm2d(mod);
}

std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<at::Tensor>> Lower(const torch::jit::script::Module& mod,
                                                                            std::string method_name) {
    LowerModule(mod);
    auto g = mod.get_method(method_name).graph();
    // Go through PyTorch Lowering to simplify graph and extract weight parameters
    auto graph_and_parameters = torch::jit::LowerGraph(*g, mod._ivalue());

    g = graph_and_parameters.first;

    // Go through TRTorch Lowering to reformat graph to be conversion friendly
    // and also segment for accelerators and executors (TRT-DLA, TRT-GPU, PYT)
    lowering::LowerGraph(g);
    // Is this necessary?
    lowering::LowerBlock(g->block());
    return graph_and_parameters;
}


} // namespace lowering
} // namespace core
} // namespace trtorch
