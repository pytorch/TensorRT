#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
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
  TORCHTRT_CHECK(
      vals.size() == specs.size(),
      "Expected dimension specifications for all input tensors"
          << ", but found " << vals.size() << " input tensors and " << specs.size() << " dimension specs");

  std::unordered_map<const torch::jit::Value*, core::ir::Input> a;
  for (size_t i = 0; i < vals.size(); i++) {
    LOG_DEBUG("Paring " << i << ": " << vals[i]->debugName() << " : " << specs[i]);
    a.insert({vals[i], specs[i]});
  }
  return a;
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

c10::optional<at::ScalarType> get_value_first_calc_dtype_opt(torch::jit::Block* b, torch::jit::Value* in) {
  TORCHTRT_ASSERT(in->owningGraph() == b->owningGraph(), "Provided input is not part of the provided graph");
  c10::optional<at::ScalarType> dtype = {};

  auto b_ins = b->inputs();
  std::unordered_set<torch::jit::Value*> b_in_set(b_ins.begin(), b_ins.end());

  TORCHTRT_ASSERT(
      in->type() == c10::TensorType::get(), "Input is not a tensor, cannot check for dtype based on calculation");

  auto consumers = in->uses();
  auto search_list = std::vector<torch::jit::Use>(consumers.begin(), consumers.end());

  for (auto iter = search_list.begin(); iter != search_list.end(); ++iter) {
    auto n = iter->user;
    LOG_GRAPH("Node we are looking at: " << util::node_info(n));
    auto ins = n->inputs();
    auto outs = n->outputs();

    bool outputs_tensor = false;
    for (auto o : outs) {
      if (o->type() == c10::TensorType::get()) {
        outputs_tensor = true;
        break;
      }
    }

    if (!outputs_tensor) {
      LOG_GRAPH("Node " << util::node_info(n) << " does not output a tensor, skipping");
      continue;
    }

    LOG_GRAPH("Node " << util::node_info(n) << " outputs a tensor");

    // If all input tensors are block inputs then this node will not give us useful type info so move to the next one
    bool all_n_ins_are_b_ins = true;
    for (auto in : ins) {
      if (b_in_set.find(in) == b_in_set.end()) {
        all_n_ins_are_b_ins = false;
        break;
      }
    }

    if (all_n_ins_are_b_ins) {
      LOG_GRAPH(
          "All inputs to Node " << util::node_info(n) << " are graph inputs, cannot be used to determine input type");
      for (auto o : outs) {
        if (o->type() == c10::TensorType::get()) {
          auto o_uses = o->uses();
          int offset = iter - search_list.begin();
          search_list.insert(search_list.end(), o_uses.begin(), o_uses.end());
          iter = search_list.begin() + offset;
        }
      }
      continue;
    }

    // If node outputs a Tensor it might be a result of tensor calcuation so check to see
    // if any inputs to the calculation can give us hints
    c10::optional<torch::jit::Node*> const_tensor_n = {};

    // Backtrace to constants which will immediately give us the Tensor type if possible
    for (auto in : ins) {
      LOG_GRAPH("Input to node: " << util::node_info(in->node()));
      if (in->type()->isSubtypeOf(torch::jit::TensorType::get())) {
        LOG_GRAPH("Input outputs a Tensor");
        if (in->node()->kind() == torch::jit::prim::Constant) {
          LOG_GRAPH("Input is a constant");
          auto const_val = in->node()->t(c10::attr::value);
          LOG_GRAPH("Found that constant tensor has type: " << const_val.scalar_type());
          dtype = {const_val.scalar_type()};
          goto exit_first_calc_dtype;
        }
      }
    }

    // Add all tensor outputs to search list if we still dont know
    for (auto o : outs) {
      if (o->type() == c10::TensorType::get()) {
        auto o_uses = o->uses();
        int offset = iter - search_list.begin();
        search_list.insert(search_list.end(), o_uses.begin(), o_uses.end());
        iter = search_list.begin() + offset;
      }
    }
  }
exit_first_calc_dtype:
  if (dtype) {
    LOG_GRAPH("Estimated input type is " << dtype.value());
  } else {
    LOG_GRAPH("Cannot determine input types from graph");
  }
  return dtype;
}

TypeMap get_block_first_calc_dtypes_opt(torch::jit::Block* b) {
  TypeMap types;

  for (auto i : b->inputs()) {
    if (i->type() == c10::TensorType::get()) {
      torch::jit::Value* in = i;
      types.insert({in, get_value_first_calc_dtype_opt(b, i)});
    }
  }
  return types;
}

} // namespace ir
} // namespace core
} // namespace torch_tensorrt
