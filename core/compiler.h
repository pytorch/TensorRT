#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "core/conversion/conversion.h"
#include "core/ir/ir.h"
#include "core/lowering/lowering.h"
#include "core/partitioning/partitioning.h"
#include "core/runtime/runtime.h"
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {

struct CompileSpec {
  CompileSpec(std::vector<ir::Input> inputs) : graph_inputs(inputs) {}
  CompileSpec(torch::jit::IValue& input_signature) : graph_inputs(input_signature) {}
  ir::GraphInputs graph_inputs;
  conversion::ConversionInfo convert_info;
  lowering::LowerInfo lower_info;
  partitioning::PartitionInfo partition_info;
};

bool CheckMethodOperatorSupport(const torch::jit::script::Module& mod, std::string method_name);

std::string ConvertGraphToTRTEngine(const torch::jit::script::Module& mod, std::string method_name, CompileSpec cfg);

torch::jit::script::Module CompileGraph(const torch::jit::script::Module& module, CompileSpec cfg);

torch::jit::script::Module EmbedEngineInNewModule(const std::string& engine, runtime::CudaDevice cuda_device);

void set_device(const int gpu_id);

} // namespace core
} // namespace torch_tensorrt
