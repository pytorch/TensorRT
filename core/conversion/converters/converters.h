#pragma once

#include <map>
#include <string>

#include "ATen/core/function_schema.h"
#include "torch/csrc/jit/runtime/custom_operator.h"

#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/conversion/converters/Weights.h"
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/var/Var.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {

typedef std::vector<Var> args;
typedef std::function<bool(ConversionCtx*, const torch::jit::Node*, args&)> OpConverter;
struct ConversionPattern {
  std::string signature;
  OpConverter converter;
};

void register_node_converter(torch::jit::FunctionSchema* signature, OpConverter& converter);
void register_node_converter(std::string signature, OpConverter& converter);
void register_node_converter(ConversionPattern p);

class RegisterNodeConversionPatterns {
 public:
  RegisterNodeConversionPatterns() = default;
  RegisterNodeConversionPatterns(const RegisterNodeConversionPatterns&) = delete;
  RegisterNodeConversionPatterns& operator=(const RegisterNodeConversionPatterns&) = delete;
  RegisterNodeConversionPatterns(RegisterNodeConversionPatterns&&) noexcept;
  RegisterNodeConversionPatterns& operator=(RegisterNodeConversionPatterns&&) noexcept;
  RegisterNodeConversionPatterns&& pattern(ConversionPattern p) &&;
};

bool node_is_convertable(const torch::jit::Node* n);
OpConverter get_node_converter_for(const torch::jit::FunctionSchema* signature);
std::vector<std::string> get_converter_list();

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
