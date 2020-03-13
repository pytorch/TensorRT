#pragma once
#include <memory>
#include "torch/csrc/jit/ir.h"

namespace trtorch {
namespace core {
namespace lowering {
    
void LowerBlock(torch::jit::Block* b);
void LowerGraph(std::shared_ptr<torch::jit::Graph>& g);

} // namespace lowering
} // namespace core
} // namespace trtorch
