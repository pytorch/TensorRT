#include <unordered_map>

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/constants.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/List.h"
#include "ATen/core/stack.h"

#include "core/util/prelude.h"
#include "core/conversion/evaluators/evaluators.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {
namespace {
using EvaluatorLUT = std::unordered_map<torch::jit::NodeKind, NodeEvaluator>;

class NodeEvaluatorRegistry {
public:
    void RegisterEvaluator(torch::jit::NodeKind node_kind, NodeEvaluator& evaluator) {
        LOG_DEBUG("Registering evaluator for " << node_kind.toQualString());
        evaluator_lut_[node_kind] = std::move(evaluator);
    }

    NodeEvaluator GetEvaluator(const torch::jit::NodeKind node_kind) {
        auto iter = evaluator_lut_.find(node_kind);
        if (iter == evaluator_lut_.end()) {
            LOG_ERROR("Requested evaluator for " << node_kind.toQualString() << ", but no such evaluator was found");
            return nullptr;
        }
        return iter->second;
    }

    bool EvalAtConversionTime(const torch::jit::Node* n) {
        auto eval_at_conversion = evaluator_lut_.find(n->kind());
        if (eval_at_conversion == evaluator_lut_.end()) {
            return false;
        } else {
            return true;
        }
    }
    
private:
    EvaluatorLUT evaluator_lut_;
};

NodeEvaluatorRegistry& get_evaluator_registry() {
    static NodeEvaluatorRegistry evaluator_registry;
    return evaluator_registry;
}
} // namespace


bool shouldEvalAtConversionTime(const torch::jit::Node* n) {
    return get_evaluator_registry().EvalAtConversionTime(n);
}

c10::optional<torch::jit::IValue> EvalNode(const torch::jit::Node* n, const kwargs& args) {
    auto evaluator = get_evaluator_registry().GetEvaluator(n->kind());
    return evaluator(n, args);
}

void register_node_evaluator(torch::jit::NodeKind node_kind, NodeEvaluator evaluator) {
    get_evaluator_registry().RegisterEvaluator(node_kind, evaluator);
}

void register_node_evaluator(EvalRegistration r) {
    register_node_evaluator(r.kind, r.evaluator);
}

RegisterNodeEvaluators&& RegisterNodeEvaluators::evaluator(EvalRegistration r) && {
    register_node_evaluator(std::move(r));
    return std::move(*this);
}

RegisterNodeEvaluators::RegisterNodeEvaluators(RegisterNodeEvaluators&&) noexcept = default;
RegisterNodeEvaluators& RegisterNodeEvaluators::RegisterNodeEvaluators::operator=(RegisterNodeEvaluators&&) noexcept = default;
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
