#include "core/util/jit_util.h"

namespace trtorch {
namespace core {
namespace util {

c10::optional<at::ScalarType> getBlockFirstCalcDType(const std::shared_ptr<torch::jit::Block>& b) {
  auto ns = b->nodes();

  c10::optional<at::ScalarType> dtype = {};

  // For each node check the inputs to find a prim:Constant, which will provide a static tensor.
  // Use that tensor to determine operating dtype for the first calculation in the block
  for (auto n : ns) {
    if (n->kind() == torch::jit::prim::Constant) {
      // Not really helpful to evaluate typing for constants
      continue;
    }

    auto ins = n->inputs();
    auto outs = n->outputs();

    bool outputs_tensor = false;
    for (auto o : outs) {
      if (o->type() == c10::TensorType::get()) {
        outputs_tensor = true;
      }
    }

    if (outputs_tensor) {
      // If all input tensors are block inputs then this node will not give us useful type info so move to the next one
      std::unordered_set<torch::jit::Value*> node_input_set = {ins.begin(), ins.end()};

      bool all_n_ins_are_b_ins = true;
      for (auto b_in : b->inputs()) {
        if (node_input_set.find(b_in) == node_input_set.end()) {
          all_n_ins_are_b_ins = false;
        }
      }

      if (all_n_ins_are_b_ins) {
        continue;
      }


      // If node outputs a Tensor it might be a result of tensor calcuation so check to see
      // if any inputs to the calculation can give us hints
      c10::optional<torch::jit::Node*> const_tensor_n = {};

      // Backtrace to constants which will immediately give us the Tensor type if possible
      for (auto in : ins) {
        if (in->type() == c10::TensorType::get()) {
          if (in->node()->kind() == torch::jit::prim::Constant) {
            auto const_ival = in->node()->get(c10::Symbol::attr("value"));
            dtype = {const_ival.value().toTensor().scalar_type()};
            goto exit_first_calc_dtype;
          }
        }
      }
    }
  }

exit_first_calc_dtype:
  return dtype;
}

} // namespace util
} // namespace core
} // namespace trtorch
