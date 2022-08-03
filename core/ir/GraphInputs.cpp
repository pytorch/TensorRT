#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace ir {

void flatten_dfs(
    std::vector<torch_tensorrt::core::ir::Input>& flattened_inputs,
    std::vector<std::vector<torch_tensorrt::core::ir::Input>>& collection_inputs,
    torch::jit::IValue input_ivalue,
    int level,
    int index) {
  if (input_ivalue.isTuple()) {
    auto input_tuple = input_ivalue.toTuple();
    int idx = 0;
    if (level == 0) {
      collection_inputs.resize(input_tuple->elements().size());
    }
    for (auto item : input_tuple->elements()) {
      torch::jit::IValue converted_item;
      int cur_idx = level < 1 ? idx : index;
      flatten_dfs(flattened_inputs, collection_inputs, item, level + 1, cur_idx);
      idx++;
    }
  } else if (input_ivalue.isList()) {
    auto input_list = input_ivalue.toList().vec();
    if (level == 0) {
      collection_inputs.resize(input_list.size());
    }
    c10::TypePtr type = input_list[0].type();
    auto converted_elements = c10::impl::GenericList(type);
    int idx = 0;
    for (auto item : input_list) {
      int cur_idx = level < 1 ? idx : index;
      flatten_dfs(flattened_inputs, collection_inputs, item, level + 1, cur_idx);
      idx++;
    }
  } else if (input_ivalue.isCustomClass()) {
    torch_tensorrt::core::ir::Input cur_input = *(input_ivalue.toCustomClass<torch_tensorrt::core::ir::Input>());
    flattened_inputs.push_back(cur_input);
    if (level == 0) { // a single value like A
      collection_inputs.resize(1);
      collection_inputs[0].push_back(cur_input);
    } else if (level == 1) { // like A in [A, A] or [(B, B), A]
      collection_inputs[index].push_back(cur_input);
    } else if (level == 2) { // like A in [(A, A), C]
      collection_inputs[index].push_back(cur_input);
    } else { // only support 2 level
      LOG_ERROR(
          "Input nesting depth exceeds currently supported depth (3), use 1 level: [A, B], or 2 level: [A, (B, C)]");
    }
  }
}

GraphInputs::GraphInputs(std::vector<ir::Input> inputs_) {
  inputs = inputs_;
  collection_inputs.resize(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); i++) {
    collection_inputs[i].push_back(inputs_[i]);
  }
}

GraphInputs::GraphInputs(torch::jit::IValue& input_signature_) {
  std::vector<torch_tensorrt::core::ir::Input> flattened_inputs;
  std::vector<std::vector<torch_tensorrt::core::ir::Input>> collection_inputs_;

  flatten_dfs(flattened_inputs, collection_inputs_, input_signature_, 0, 0);
  inputs = flattened_inputs;
  input_signature = input_signature_;
  collection_inputs = collection_inputs_;
  LOG_DEBUG("Collection Input Size: " << collection_inputs_.size());
}

} // namespace ir
} // namespace core
} // namespace torch_tensorrt