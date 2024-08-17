#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto layer_norm_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? gamma, Tensor? beta,
                           float eps, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensorOrFreeze(ctx);
      auto input_shape = input->getDimensions();
      auto input_shape_vec = util::toVec(input_shape);
      auto normalized_shape = args[1].unwrapToIntList();
      auto normalized_shape_vec = util::toVec(util::toDims(normalized_shape));
      auto axis = input_shape_vec.size() - normalized_shape_vec.size();
      uint32_t axes_mask = 0;
      for (size_t i = axis; i < input_shape_vec.size(); i++) {
        axes_mask |= 1 << i;
      }

      nvinfer1::ITensor* gamma = nullptr;
      if (args[2].IValue()->isNone()) {
        auto gamma_torch_tensor =
            torch::ones(input_shape_vec, torch::TensorOptions().dtype(util::TRTDataTypeToScalarType(input->getType())));
        gamma = tensor_to_const(ctx, gamma_torch_tensor);
      } else {
        gamma = args[2].ITensorOrFreeze(ctx);
        gamma = add_expand(ctx, gamma, input_shape);
      }

      nvinfer1::ITensor* beta = nullptr;
      if (args[3].IValue()->isNone()) {
        auto beta_torch_tensor = torch::zeros(
            input_shape_vec, torch::TensorOptions().dtype(util::TRTDataTypeToScalarType(input->getType())));
        beta = tensor_to_const(ctx, beta_torch_tensor);
      } else {
        beta = args[3].ITensorOrFreeze(ctx);
        beta = add_expand(ctx, beta, input_shape);
      }

      auto eps = args[4].unwrapToDouble();

      auto normalize_layer = ctx->net->addNormalization(*input, *gamma, *beta, axes_mask);
      TORCHTRT_CHECK(normalize_layer, "Unable to create layer_norm from node: " << *n);
      normalize_layer->setName(util::node_info(n).c_str());
      normalize_layer->setEpsilon(eps);
      normalize_layer->setComputePrecision(input->getType());
      auto normalized = normalize_layer->getOutput(0);

      ctx->AssociateValueAndTensor(n->outputs()[0], normalized);
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
