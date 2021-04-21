#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "core/conversion/conversion.h"
#include "torch/csrc/jit/api/module.h"

namespace trtorch {
namespace core {

struct CompileSpec {
  CompileSpec(std::vector<conversion::InputRange> input_ranges) : convert_info(std::move(input_ranges)) {}
  conversion::ConversionInfo convert_info;
};

bool CheckMethodOperatorSupport(const torch::jit::script::Module& mod, std::string method_name);

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod, std::string method_name, CompileSpec cfg);

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, CompileSpec cfg);

torch::jit::script::Module EmbedEngineInNewModule(const std::string& engine);

void set_device(const int gpu_id);

} // namespace core
} // namespace trtorch
