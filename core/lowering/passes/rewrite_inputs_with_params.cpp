#include "core/util/prelude.h"
#include "torch/csrc/jit/ir/constants.h"

namespace torch_tensorrt {
namespace core {
namespace lowering {
namespace passes {

void RewriteInputsWithParams(std::shared_ptr<torch::jit::Graph>& g, std::vector<torch::jit::IValue>& params) {
  auto input_size = g->inputs().size();
  auto param_it = params.rbegin();
  for (int i = input_size - 1; i >= 0; --i) {
    if (g->inputs()[i]->type() != c10::TensorType::get() &&
        g->inputs()[i]->type()->kind() != torch::jit::TypeKind::TupleType &&
        g->inputs()[i]->type()->kind() != torch::jit::TypeKind::ListType && param_it != params.rend()) {
      auto val = *param_it;
      if (val.isTensor()) {
        at::Tensor val_tensor = val.toTensor();
        if (val_tensor.requires_grad()) {
          val_tensor.set_requires_grad(false);
          val = val_tensor;
        }
      }
      auto new_constant = torch::jit::tryInsertConstant(*g, val);
      ++param_it;
      if (new_constant) {
        g->inputs()[i]->replaceAllUsesWith(*new_constant);
        g->eraseInput(i);
        // erase an iterator, should be safe
        params.erase(param_it.base());
      }
    }
  }
  LOG_GRAPH("After RewriteInputsWithParams: " << *g);
}

} // namespace passes
} // namespace lowering
} // namespace core
} // namespace torch_tensorrt
