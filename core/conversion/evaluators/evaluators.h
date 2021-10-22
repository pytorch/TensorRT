#pragma once

#include <map>
#include <set>
#include <string>

#include "torch/csrc/jit/ir/ir.h"

#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/conversion/var/Var.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace evaluators {

typedef std::map<const torch::jit::Value*, Var> kwargs;

inline bool constTypesOnly(kwargs& args) {
  std::set<Var::Type> types;
  for (auto a : args) {
    if (a.second.type() == Var::kITensor) {
      return false;
    }
  }
  return true;
}

// NOTE: The input args are a dictionary of Value -> Var this means
// inputs will not be repeated. We did this because while in the case
// of converters we have the function schema to lay out argument order,
// evaluators dont use the schema, they use node kind as key so it easier
// to use the node itself to pull out arguments.
// This means that you should iterate over node inputs vs. the args
// when writing evaluators
typedef std::function<c10::optional<torch::jit::IValue>(const torch::jit::Node*, kwargs&)> NodeEvaluator;

struct EvalOptions {
  std::set<c10::TypePtr> blacklisted_output_types;
  std::vector<c10::OperatorName> valid_schemas;
  std::vector<std::string> supported_variants;
  EvalOptions() = default;
  EvalOptions& blacklistOutputTypes(std::set<c10::TypePtr> types) {
    use_options = true;
    blacklisted_output_types = types;
    return *this;
  }
  EvalOptions& validSchemas(std::set<std::string> schemas) {
    std::copy(schemas.begin(), schemas.end(), std::back_inserter(supported_variants));
    use_options = true;
    for (auto s : schemas) {
      valid_schemas.push_back(torch::jit::parseSchema(s).operator_name());
    }
    return *this;
  }
  bool use() {
    return use_options;
  }

 private:
  bool use_options = false;
};

struct EvalRegistration {
  torch::jit::NodeKind kind;
  NodeEvaluator evaluator;
  EvalOptions options;
  EvalRegistration() = default;
  EvalRegistration(torch::jit::NodeKind _kind, NodeEvaluator _evaluator)
      : kind(_kind), evaluator(_evaluator), options(EvalOptions()){};
  EvalRegistration(torch::jit::NodeKind _kind, NodeEvaluator _evaluator, EvalOptions _options)
      : kind(_kind), evaluator(_evaluator), options(_options){};
};

c10::optional<torch::jit::IValue> EvalNode(const torch::jit::Node* n, kwargs& args);
bool shouldEvalAtConversionTime(const torch::jit::Node* n);
std::vector<std::string> getEvaluatorList();
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
} // namespace torch_tensorrt
