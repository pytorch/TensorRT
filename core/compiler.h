#pragma once

#include <vector>
#include "torch/csrc/jit/api/module.h"
#include "core/conversion/conversion.h"

namespace trtorch {
namespace core {

struct ExtraInfo {
    ExtraInfo(std::vector<conversion::InputRange> input_ranges)
        : convert_info(std::move(input_ranges)) {}
    conversion::ConversionInfo convert_info;
};

bool CheckMethodOperatorSupport(const torch::jit::script::Module& mod, std::string method_name);

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod,
                                    std::string method_name, ExtraInfo cfg);

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, ExtraInfo cfg);

} // namespace core
} // namespace trtorch
