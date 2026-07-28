#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/csrc/jit/frontend/function_schema_parser.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {

std::string canonical_schema_string(const torch::jit::FunctionSchema& schema) {
  std::ostringstream out;

  out << schema.name();
  out << "(";

  bool seen_kwarg_only = false;
  for (size_t i = 0; i < schema.arguments().size(); ++i) {
    if (i > 0)
      out << ", ";
    if (schema.arguments()[i].kwarg_only() && !seen_kwarg_only) {
      out << "*, ";
      seen_kwarg_only = true;
    }
    const auto& arg = schema.arguments()[i];
    out << arg.type()->str() << " " << arg.name();
  }

  out << ") -> ";
  if (schema.returns().size() == 1) {
    out << schema.returns().at(0).type()->str();
  } else if (schema.returns().size() > 1) {
    out << "(";
    for (size_t i = 0; i < schema.returns().size(); ++i) {
      if (i > 0)
        out << ", ";
      out << schema.returns()[i].type()->str();
    }
    out << ")";
  }
  return out.str();
}

namespace {
using ConverterLUT = std::unordered_map<c10::OperatorName, OpConverter>;

class NodeConverterRegistry {
 public:
  bool RegisterConverter(torch::jit::FunctionSchema* signature, OpConverter& converter) {
    LOG_DEBUG("Registering converter for " << canonical_schema_string(*signature));
    registered_converter_schemas_.insert(c10::toString(*signature));
    auto name = signature->operator_name();
    auto iter = converter_lut_.find(name);
    if (iter != converter_lut_.end()) {
      LOG_WARNING("Overriding already registered converter " << signature->name() << ", unexpected behavior may occur");
    }
    converter_lut_[name] = std::move(converter);
    return true;
  }

  OpConverter GetConverter(const torch::jit::FunctionSchema* signature) {
    auto name = signature->operator_name();
    auto iter = converter_lut_.find(name);
    if (iter == converter_lut_.end()) {
      LOG_ERROR("Requested converter for " << signature->name() << ", but no such converter was found");
      // ASK: Is there a better way than returning a nullptr?
      return nullptr;
    }
    return iter->second;
  }

  bool Convertable(const torch::jit::Node* n) {
    auto schema = n->maybeSchema();
    if (schema) {
      auto name = schema->operator_name();
      auto iter = converter_lut_.find(name);
      if (iter == converter_lut_.end()) {
        return false;
      } else {
        return true;
      }
    } else {
      LOG_DEBUG("Unable to get schema for Node " << util::node_info(n) << " (NodeConverterRegistry.Convertable)");
      return false;
    }
  }

  std::vector<std::string> GetRegisteredConverterList() {
    std::vector<std::string> converter_list;
    std::copy(
        registered_converter_schemas_.begin(), registered_converter_schemas_.end(), std::back_inserter(converter_list));
    return converter_list;
  }

 private:
  ConverterLUT converter_lut_;
  std::set<std::string> registered_converter_schemas_;
};

NodeConverterRegistry& get_converter_registry() {
  static NodeConverterRegistry converter_registry;
  return converter_registry;
}
} // namespace

void register_node_converter(torch::jit::FunctionSchema* signature, OpConverter& converter) {
  get_converter_registry().RegisterConverter(signature, converter);
}

void register_node_converter(std::string signature, OpConverter& converter) {
  auto schema = torch::jit::parseSchema(signature);
  // TODO: CHECKING THIS IS A VALID SCHEMA AND QUITING IF NOT
  register_node_converter(&schema, converter);
}

void register_node_converter(ConversionPattern p) {
  register_node_converter(p.signature, p.converter);
}

OpConverter get_node_converter_for(const torch::jit::FunctionSchema* signature) {
  return get_converter_registry().GetConverter(signature);
}

bool node_is_convertable(const torch::jit::Node* n) {
  return get_converter_registry().Convertable(n);
}

std::vector<std::string> get_converter_list() {
  return get_converter_registry().GetRegisteredConverterList();
}

RegisterNodeConversionPatterns&& RegisterNodeConversionPatterns::pattern(ConversionPattern p) && {
  register_node_converter(std::move(p));
  return std::move(*this);
}

RegisterNodeConversionPatterns::RegisterNodeConversionPatterns(RegisterNodeConversionPatterns&&) noexcept = default;
RegisterNodeConversionPatterns& RegisterNodeConversionPatterns::RegisterNodeConversionPatterns::operator=(
    RegisterNodeConversionPatterns&&) noexcept = default;

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
