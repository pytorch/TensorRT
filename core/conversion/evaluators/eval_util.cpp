#include "ATen/core/ivalue.h"
#include "ATen/core/List.h"
#include "core/util/prelude.h"
#include "ATen/core/functional.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {

//TODO: Switch back to PyTorch canonical implimentation
c10::optional<torch::jit::IValue> toIValue(const torch::jit::Value* v) {
  if (v->node()->kind() != torch::jit::prim::Constant || v->type()->cast<c10::FunctionType>()) {
    return c10::nullopt;
  }
  const torch::jit::Node* node = v->node();
  const c10::TypePtr& type = v->type();
  if (type->isSubtypeOf(c10::TensorType::get())) {
    return node->t(c10::attr::value);
  } else if (type->isSubtypeOf(c10::BoolType::get())) {
    return (bool)node->i(c10::attr::value);
  } else if (
    type->isSubtypeOf(c10::NumberType::get()) &&
     node->kindOf(c10::attr::value) == torch::jit::AttributeKind::i) {
    return node->i(c10::attr::value);
  } else if (
    type->isSubtypeOf(c10::NumberType::get()) &&
    node->kindOf(c10::attr::value) == torch::jit::AttributeKind::f) {
    return node->f(c10::attr::value);
  } else if (type->isSubtypeOf(c10::ListType::ofInts())) {
    try {
      const auto& is = node->is(c10::attr::value);
      return is;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->isSubtypeOf(c10::ListType::ofFloats())) {
    try {
      const auto& fs = node->fs(c10::attr::value);
      return fs;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->isSubtypeOf(c10::ListType::ofBools())) {
    const auto bs = c10::fmap<bool>(node->is(c10::attr::value));
    return bs;
  } else if (type->isSubtypeOf(c10::ListType::ofTensors())) {
    try {
      const auto& ts = node->ts(c10::attr::value);
      return ts;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->isSubtypeOf(c10::ListType::ofStrings())) {
    try {
      const auto& ss = node->ss(c10::attr::value);
      auto vals = c10::impl::GenericList(c10::StringType::get());
      for (const auto& str : ss) {
        vals.push_back(str);
      }
      return vals;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (
      type->cast<c10::ListType>() &&
      node->kindOf(c10::attr::value) == torch::jit::AttributeKind::ival) {
    const auto& list = node->ival(c10::attr::value);
    TRTORCH_ASSERT(list.isList(), "Is not a list");
    return list;
  } else if (
      type->cast<c10::DictType>() &&
      node->kindOf(c10::attr::value) == torch::jit::AttributeKind::ival) {
    const auto& dict = node->ival(c10::attr::value);
    TRTORCH_ASSERT(dict.isGenericDict(), "Is not a dict");
    return dict;
  } else if (
      type->cast<c10::TupleType>() &&
      node->kindOf(c10::attr::value) == torch::jit::AttributeKind::ival) {
    const auto& tup = node->ival(c10::attr::value);
    TRTORCH_ASSERT(tup.isTuple(), "Is not a tuple");
    return tup;
  } else if (type == c10::StringType::get()) {
    const auto& s = node->s(c10::attr::value);
    return s;
  } else if (type == c10::DeviceObjType::get()) {
    auto d = c10::Device(node->s(c10::attr::value));
    return d;
  } else if (node->mustBeNone()) {
    return torch::jit::IValue();
  } else {
    std::stringstream ss;
    ss << "constant literal not supported for: " << type->str();
    throw std::runtime_error(ss.str());
  }
}

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
