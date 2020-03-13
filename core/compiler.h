#pragma once

#include <vector>
#include "torch/csrc/jit/script/module.h"
#include "core/conversion/conversion.h"

namespace trtorch {
namespace core {

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod,
                                    std::string method_name, conversion::ExtraInfo cfg);
torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, conversion::ExtraInfo cfg);

} // namespace core
} // namespace trtorch 
