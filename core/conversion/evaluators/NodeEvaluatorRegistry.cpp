#include <unordered_map>

#include "ATen/core/List.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/stack.h"
#include "torch/csrc/jit/ir/constants.h"
#include "torch/csrc/jit/ir/ir.h"

#include "core/conversion/evaluators/evaluators.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace evaluators {
namespace {
using EvaluatorLUT = std::unordered_map<torch::jit::NodeKind, EvalRegistration>;

bool FindInVec(std::vector<c10::OperatorName>& names, c10::OperatorName target) {
  for (auto n : names) {
    if (n == target) {
      return true;
    }
  }
  return false;
}

class NodeEvaluatorRegistry {
 public:
  void RegisterEvaluator(torch::jit::NodeKind node_kind, EvalRegistration eval_reg) {
    LOG_DEBUG("Registering evaluator for " << node_kind.toQualString());
    auto iter = evaluator_lut_.find(node_kind);
    if (iter != evaluator_lut_.end()) {
      TORCHTRT_THROW_ERROR(
          "Attempting to override already registered evaluator " << node_kind.toQualString()
                                                                 << ", merge implementations instead");
    }
    for (auto const& e : eval_reg.options.supported_variants) {
      registered_evaluator_schemas_.insert(e);
    }
    evaluator_lut_[node_kind] = std::move(eval_reg);
  }

  NodeEvaluator FindEvaluator(const torch::jit::Node* n) {
    auto node_kind = n->kind();
    auto iter = evaluator_lut_.find(node_kind);
    if (iter == evaluator_lut_.end()) {
      return nullptr;
    }
    auto eval_reg = iter->second;
    if (eval_reg.options.use()) {
      for (auto o : n->outputs()) {
        if (eval_reg.options.blacklisted_output_types.find(o->type()) !=
            eval_reg.options.blacklisted_output_types.end()) {
          return nullptr;
        }
      }

      if (eval_reg.options.valid_schemas.size() != 0) {
        auto schema = n->maybeSchema();
        TORCHTRT_CHECK(
            schema,
            "Evaluator for " << node_kind.toQualString() << " only runs on certain schemas, but schema for target"
                             << " node is not a supported schema variant of " << node_kind.toQualString());
        if (!FindInVec(eval_reg.options.valid_schemas, schema->operator_name())) {
          return nullptr;
        }
      }
    }

    return eval_reg.evaluator;
  }

  NodeEvaluator GetEvaluator(const torch::jit::Node* n) {
    auto evaluator = FindEvaluator(n);
    TORCHTRT_CHECK(
        evaluator, "Requested evaluator for " << n->kind().toQualString() << ", but no such evaluator was found");
    return evaluator;
  }

  std::vector<std::string> GetRegisteredEvaluatorList() {
    std::vector<std::string> evaluator_list;
    std::copy(
        registered_evaluator_schemas_.begin(), registered_evaluator_schemas_.end(), std::back_inserter(evaluator_list));
    return evaluator_list;
  }

  bool EvalAtConversionTime(const torch::jit::Node* n) {
    auto evaluator = FindEvaluator(n);
    if (evaluator == nullptr) {
      return false;
    } else {
      return true;
    }
  }

 private:
  EvaluatorLUT evaluator_lut_;
  std::set<std::string> registered_evaluator_schemas_;
};

NodeEvaluatorRegistry& get_evaluator_registry() {
  static NodeEvaluatorRegistry evaluator_registry;
  return evaluator_registry;
}
} // namespace

bool shouldEvalAtConversionTime(const torch::jit::Node* n) {
  return get_evaluator_registry().EvalAtConversionTime(n);
}

std::vector<std::string> getEvaluatorList() {
  return get_evaluator_registry().GetRegisteredEvaluatorList();
}

c10::optional<torch::jit::IValue> EvalNode(const torch::jit::Node* n, kwargs& args) {
  auto evaluator = get_evaluator_registry().GetEvaluator(n);
  return evaluator(n, args);
}

void register_node_evaluator(torch::jit::NodeKind node_kind, EvalRegistration eval_reg) {
  get_evaluator_registry().RegisterEvaluator(node_kind, std::move(eval_reg));
}

void register_node_evaluator(EvalRegistration r) {
  register_node_evaluator(r.kind, std::move(r));
}

RegisterNodeEvaluators&& RegisterNodeEvaluators::evaluator(EvalRegistration r) && {
  register_node_evaluator(std::move(r));
  return std::move(*this);
}

RegisterNodeEvaluators::RegisterNodeEvaluators(RegisterNodeEvaluators&&) noexcept = default;
RegisterNodeEvaluators& RegisterNodeEvaluators::RegisterNodeEvaluators::operator=(RegisterNodeEvaluators&&) noexcept =
    default;
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
