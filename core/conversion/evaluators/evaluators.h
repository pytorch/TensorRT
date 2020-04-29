#pragma once

#include <string>
#include <map>

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {

typedef std::map<const torch::jit::Value*, const torch::jit::IValue*> kwargs;

// NOTE: The input args are a dictionary of Value -> IValue this means
// inputs will not be repeated. We did this so writing encoders
// is similar to converters (where a dictionary makes more sense)
// This mean that you should iterate over node inputs vs. the args
// when writing evaluators
typedef std::function<c10::optional<torch::jit::IValue>(const torch::jit::Node*, const kwargs&)> NodeEvaluator;

struct EvalRegistration {
    torch::jit::NodeKind kind;
    NodeEvaluator evaluator;
};

c10::optional<torch::jit::IValue> EvalNode(const torch::jit::Node* n, const kwargs& args);
bool shouldEvalAtConversionTime(const torch::jit::Node* n);
void register_node_evaluator(torch::jit::NodeKind node_kind, NodeEvaluator evaluator);
void register_node_evaluator(EvalRegistration r);

class RegisterNodeEvaluators {
public:
    RegisterNodeEvaluators() = default;
    RegisterNodeEvaluators(const RegisterNodeEvaluators&) = delete;
    RegisterNodeEvaluators& operator=(const RegisterNodeEvaluators&) = delete;
    RegisterNodeEvaluators(RegisterNodeEvaluators&&) noexcept;
    RegisterNodeEvaluators& operator=(RegisterNodeEvaluators&&) noexcept;
    RegisterNodeEvaluators&& evaluator(EvalRegistration r) &&;
};

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
