#include <string>
#include <unordered_set>

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace conversion {

// clang-format off
const std::unordered_set<std::string>& get_non_convertable_nodes() {
  // Set of nodes that should not invoke a converter or evaluator
  static std::unordered_set<std::string> nonconvertable_nodes = {
    "aten::manual_seed",
    "aten::grad",
    "aten::backward",
    "aten::save",
    "aten::contiguous",
    "aten::to",
    "prim::RaiseException",
    "prim::Print",
    "prim::device",
    "prim::GetAttr",
    "prim::CallMethod",
    "prim::Drop",
    "aten::dropout",
    "aten::dropout_"};
  return nonconvertable_nodes;
}
// clang-format on

bool isNodeConversionIgnored(const torch::jit::Node* n) {
  auto kind = n->kind();
  auto convertableIt = get_non_convertable_nodes().find(kind.toQualString());
  if (convertableIt == get_non_convertable_nodes().end()) {
    return false;
  } else {
    return true;
  }
}

} // namespace conversion
} // namespace core
} // namespace trtorch
