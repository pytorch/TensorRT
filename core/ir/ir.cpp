#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace ir {

InputSpecMap associate_specs_with_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    std::vector<Input> specs,
    StaticParams& static_params) {
  auto tensor_inputs = get_tensor_inputs(g, static_params);
  return pair_input_vals_with_specs(tensor_inputs, specs);
}

InputSpecMap pair_input_vals_with_specs(std::vector<const torch::jit::Value*> vals, std::vector<Input> specs) {
  TRTORCH_CHECK(
      vals.size() == specs.size(),
      "Expected dimension specifications for all input tensors"
          << ", but found " << vals.size() << " input tensors and " << specs.size() << " dimension specs");

  std::unordered_map<const torch::jit::Value*, core::ir::Input> a;
  for (size_t i = 0; i < vals.size(); i++) {
    LOG_DEBUG("Paring " << i << ": " << vals[i]->debugName() << " : " << specs[i]);
    a.insert({vals[i], specs[i]});
  }
  return std::move(a);
}

std::vector<const torch::jit::Value*> get_tensor_inputs(
    std::shared_ptr<torch::jit::Graph>& g,
    StaticParams& static_params) {
  std::vector<const torch::jit::Value*> input_tensors;
  auto inputs = g->inputs();
  for (auto in : inputs) {
    // Disregarding inputs that are not tensors or are static
    //
    // Ex.
    // self.1:__torch__.alexnet -> ignored
    // input.1:Tensor -> used
    if (in->type()->isSubtypeOf(c10::TensorType::get()) && static_params.find(in) == static_params.end()) {
      input_tensors.push_back(in);
    }
  }
  return input_tensors;
}

} // namespace ir
} // namespace core
} // namespace trtorch