#pragma once
#include <memory>
#include "core/ir/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {

struct LowerInfo {
  // Internal flag to ensure torch.jit.Module does not get freezed in lowering.cpp. This is required for QAT models.
  bool unfreeze_module = false;
  // CommonSubexpressionElimination removes duplicate expressions which are used frequently in the graph.
  // for eg:  CSE replaces similar value-d stride nodes of multiple conv layers in a network with a single stride node.
  // In QAT models, if two conv layers are consuming same input, there is a QDQ node for each input of the conv.
  // Since these QDQ nodes will be identical as they share same input, one of them is eliminated due to CSE lowering
  // pass. Disable this in order to not disturb TensorRT's QAT optimizations.
  bool disable_cse = false;

  // Whether the originating caller is `convert_method_to_trt_engine` (true) or `compile` (false)
  bool converting_to_trt_engine = false;

  ir::Device target_device;
  std::vector<std::string> forced_fallback_modules;
  friend std::ostream& operator<<(std::ostream& os, const LowerInfo& l);

  std::string getGPUDeviceString() const {
    return "cuda:" + std::to_string(target_device.gpu_id);
  };
};

void LowerBlock(torch::jit::Block* b);
void LowerGraph(std::shared_ptr<torch::jit::Graph>& g, LowerInfo lower_info);
int AutocastLongInputs(
    std::shared_ptr<torch::jit::Graph>& g,
    ir::TypeMap input_type_map,
    std::string target_device_name);
torch::jit::Module LowerModule(
    const torch::jit::Module& mod,
    std::string method_name,
    std::unordered_set<std::string> forced_fallback_modules);
std::pair<std::shared_ptr<torch::jit::Graph>, std::vector<torch::jit::IValue>> Lower(
    const torch::jit::Module& mod,
    std::string method_name,
    const LowerInfo& lower_info);

} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
