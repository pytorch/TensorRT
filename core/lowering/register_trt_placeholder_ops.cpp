#include <limits>
#include "torch/csrc/jit/runtime/custom_operator.h"

namespace torch {
namespace jit {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

RegisterOperators trt_placeholder_ops_reg({
    /// Op marks a Tensor to be conveted from an Torch Tensor
    /// to a TRT constant Tensor
    Operator(
        "trt::const(Tensor val) -> Tensor",
        [](Stack& stack) { /*noop*/ },
        aliasAnalysisFromSchema()),
    Operator(
        "trt::attn_bias_from_attn_mask(Tensor attn_mask) -> Tensor",
        [](Stack& stack) {
          auto attn_mask = pop(stack).to<at::Tensor>();
          if (attn_mask.scalar_type() == at::kBool) {
            attn_mask.masked_fill_(attn_mask.logical_not(), -std::numeric_limits<float>::infinity());
          }
          return attn_mask;
        },
        c10::AliasAnalysisKind::CONSERVATIVE),
});

} // namespace jit
} // namespace torch
