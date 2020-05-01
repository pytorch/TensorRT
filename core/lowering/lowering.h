#pragma once
#include <memory>
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace lowering {

void LowerBlock(torch::jit::Block* b);
void LowerGraph(std::shared_ptr<torch::jit::Graph>& g);
torch::jit::Module LowerModule(const torch::jit::script::Module& mod);
std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<at::Tensor>> Lower(const torch::jit::script::Module& mod,
                                                                             std::string method_name);

} // namespace lowering
} // namespace core
} // namespace trtorch
