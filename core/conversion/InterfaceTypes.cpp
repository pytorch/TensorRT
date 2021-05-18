#include <set>

#include "core/conversion/conversion.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {

GraphParams get_named_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<torch::jit::IValue> params) {
  GraphParams named_params;
  auto param_it = params.begin();
  for (auto in : inputs) {
    if (in->type() != c10::TensorType::get() && param_it != params.end()) {
      named_params[in] = *param_it;
      ++param_it;
    }
  }

  TRTORCH_CHECK(
      named_params.size() == params.size(),
      "Graph parameter parsing failed, mismatched number of static parameters and IValues")
  return std::move(named_params);
}

} // namespace conversion
} // namespace core
} // namespace trtorch
