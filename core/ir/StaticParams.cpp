#include <set>

#include "core/ir/ir.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace ir {

StaticParams get_static_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<torch::jit::IValue> params) {
  StaticParams static_params;
  auto param_it = params.begin();
  for (auto in : inputs) {
    if (in->type() != c10::TensorType::get() && param_it != params.end()) {
      static_params[in] = *param_it;
      ++param_it;
    }
  }

  TRTORCH_CHECK(
      static_params.size() == params.size(),
      "Graph parameter parsing failed, mismatched number of static parameters and IValues")
  return std::move(static_params);
}

} // namespace ir
} // namespace core
} // namespace trtorch
