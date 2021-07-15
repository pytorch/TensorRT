#pragma once
#include <memory>
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace lowering {

struct LowerInfo {
	std::vector<std::string> forced_fallback_modules;
}

void LowerBlock(torch::jit::Block* b);
void LowerGraph(std::shared_ptr<torch::jit::Graph>& g);
torch::jit::Module LowerModule(const torch::jit::script::Module& mod);
void NotateModuleForFallback(const torch::jit::script::Module& mod, std::string method_name, std::unordered_set<std::string> forced_fallback_modules);
std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>> Lower(
    const torch::jit::script::Module& mod,
    std::string method_name, const LowerInfo& lower_info);

} // namespace lowering
} // namespace core
} // namespace trtorch
