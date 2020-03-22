#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"
#include "torch/csrc/jit/script/function_schema_parser.h"

namespace trtorch {
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
using ConverterLUT = std::unordered_map<torch::jit::Symbol, OpConverter>;

class NodeConverterRegistry {
public:
    bool RegisterConverter(torch::jit::FunctionSchema* signature, OpConverter& converter) {
        // NOTE: This is useful for people developing extentions to the conversion registry as is
        // If you are working on the core conversion library and the conversion registry
        // itself, it might helpful to set -DDEBUG_MSGS when you compile so you can watch the
        // registration of core converters during init, otherwise the messages will be masked
        LOG_DEBUG("Registering Converter for " << canonical_schema_string(*signature));
        auto sym = torch::jit::Symbol::fromQualString(signature->name());
        converter_lut_[sym] = std::move(converter);
        return true;
    }

    OpConverter GetConverter(const torch::jit::FunctionSchema* signature) {
        auto sym = torch::jit::Symbol::fromQualString(signature->name());
        auto iter = converter_lut_.find(sym);
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
            auto converter = GetConverter(schema);
            if (converter) {
                return true;
            } else {
                LOG_DEBUG("Node has no registered converter: " << util::node_info(n) \
                          << " (NodeConverterRegistry.Convertable)\nSchema: " << *schema);
                return false;
            }
        } else {
            LOG_DEBUG("Unable to get schema for Node " << util::node_info(n) \
                      << " (NodeConverterRegistry.Convertable)");
            return false;
        }
    }
    
private:
    ConverterLUT converter_lut_;
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
    
RegisterNodeConversionPatterns&& RegisterNodeConversionPatterns::pattern(ConversionPattern p) && {
    register_node_converter(std::move(p));
    return std::move(*this);
}

RegisterNodeConversionPatterns::RegisterNodeConversionPatterns(RegisterNodeConversionPatterns&&) noexcept = default;
RegisterNodeConversionPatterns& RegisterNodeConversionPatterns::RegisterNodeConversionPatterns::operator=(RegisterNodeConversionPatterns&&) noexcept = default;

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
