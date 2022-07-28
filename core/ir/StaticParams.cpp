#include <set>

#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace ir {

StaticParams get_static_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<torch::jit::IValue> params) {
  StaticParams static_params;
  auto param_it = params.begin();
  for (auto in : inputs) {
    // handle TensorType, TupleType and ListType
    if (in->type() != c10::TensorType::get() && in->type()->kind() != torch::jit::TypeKind::TupleType &&
        in->type()->kind() != torch::jit::TypeKind::ListType && param_it != params.end()) {
      static_params[in] = *param_it;
      ++param_it;
    }
  }

  TORCHTRT_CHECK(
      static_params.size() == params.size(),
      "Graph parameter parsing failed, mismatched number of static parameters and IValues")
  return static_params;
}

} // namespace ir
} // namespace core
} // namespace torch_tensorrt
