#include "torch/csrc/jit/runtime/custom_operator.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

/// Op marks a Tensor to be conveted from an Torch Tensor
/// to a TRT constant Tensor
RegisterOperators trt_const_op_reg({
  Operator(
    "trt::const(Tensor val) -> Tensor",
    [](Stack& stack) {
      return 0; //noop
    },
    aliasAnalysisFromSchema())});

} // namespace jit
} // namespace torch