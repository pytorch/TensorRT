#include <string>
#include <unordered_set>

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace conversion {

const std::unordered_set<std::string>& get_non_convertable_nodes() {
    // Set of nodes that should not invoke a converter or evaluator
    static std::unordered_set<std::string> nonconvertable_nodes = {
        "aten::manual_seed",
        "aten::grad",
        "aten::backward",
        "aten::save",
        "prim::RaiseException",
        "prim::Print",
        "prim::device",
        "prim::GetAttr",
        "prim::CallMethod",
        "aten:dropout",
    };
    return nonconvertable_nodes;
}

bool isNodeConversionBlacklisted(const torch::jit::Node* n) {
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
