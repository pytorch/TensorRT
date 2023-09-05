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
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);

      /* Layer_Norm normalizes over last N dimensions.
         normalizaed_shape could be (C,H,W), (H,W), or (W). */
      // This could be an IntList or ITensorList. We only need the size of this list.
      auto normalized_shape = args[1].IValue()->toList();

      // Unwrap eps.
      auto eps = args[4].unwrapToDouble();

      LOG_DEBUG("cudnn disregarded");

      // Set up  axis_ask for E[x].
      uint32_t axis_mask = 0;
      for (size_t i = 0; i < normalized_shape.size(); i++) {
        axis_mask |= 1 << (shape.size() - i - 1);
      }
      LOG_DEBUG("Axis Mask for E[x]" << std::bitset<32>(axis_mask));

      // E[x]
      auto mean_expected = ctx->net->addReduce(*input, nvinfer1::ReduceOperation::kAVG, axis_mask, true);
      TORCHTRT_CHECK(mean_expected, "Unable to create mean_expected from node: " << *n);
      mean_expected->setName((util::node_info(n) + "_mean_expected").c_str());
      auto mean_expected_out = mean_expected->getOutput(0);

      // X-E[x]
      auto sub = add_elementwise(
          ctx, nvinfer1::ElementWiseOperation::kSUB, input, mean_expected_out, (util::node_info(n) + "_sub").c_str());
      TORCHTRT_CHECK(sub, "Unable to create Sub layer from node: " << *n);
      sub->setName((util::node_info(n) + "_sub").c_str());
      auto xsubmean_out = sub->getOutput(0);

      // Variance = mean(pow(xsubmean,2))
      float pow_scalar = 2;
      auto exponent = tensor_to_const(ctx, torch::tensor({pow_scalar}));
      auto pow = add_elementwise(
          ctx, nvinfer1::ElementWiseOperation::kPOW, xsubmean_out, exponent, (util::node_info(n) + "_pow").c_str());
      TORCHTRT_CHECK(pow, "Unable to create Pow layer from node: " << *n);
      pow->setName((util::node_info(n) + "_pow").c_str());
      auto pow_out = pow->getOutput(0);

      auto mean_var = ctx->net->addReduce(*pow_out, nvinfer1::ReduceOperation::kAVG, axis_mask, true);
      TORCHTRT_CHECK(mean_var, "Unable to create mean_var from node: " << *n);
      mean_var->setName((util::node_info(n) + "_mean_var").c_str());
      auto mean_var_out = mean_var->getOutput(0);

      // Variance + eps
      auto eps_tensor = tensor_to_const(ctx, torch::tensor({eps}));
      auto add = add_elementwise(
          ctx, nvinfer1::ElementWiseOperation::kSUM, mean_var_out, eps_tensor, (util::node_info(n) + "_add").c_str());
      TORCHTRT_CHECK(add, "Unable to create Add layer from node: " << *n);
      add->setName((util::node_info(n) + "_add").c_str());
      auto add_out = add->getOutput(0);

      // SQRT((Var + eps))
      auto sqrt = ctx->net->addUnary(*add_out, nvinfer1::UnaryOperation::kSQRT);
      TORCHTRT_CHECK(sqrt, "Unable to create unary(sqrt) from node: " << *n);
      sqrt->setName((util::node_info(n) + "_sqrt").c_str());
      auto sqrt_out = sqrt->getOutput(0);

      // (x - E[x]) / sqrt((var + eps))
      auto div = add_elementwise(
          ctx, nvinfer1::ElementWiseOperation::kDIV, xsubmean_out, sqrt_out, (util::node_info(n) + "_div").c_str());
      TORCHTRT_CHECK(div, "Unable to create div layer from node: " << *n);
      div->setName((util::node_info(n) + "_div").c_str());
      auto div_out = div->getOutput(0);

      if (!args[2].IValue()->isTensor() && !args[3].IValue()->isTensor()) {
        ctx->AssociateValueAndTensor(n->outputs()[0], div_out);
        return true;
      }

      auto normalized = div_out;

      // gamma
      if (args[2].IValue()->isTensor()) {
        auto gamma = args[2].ITensorOrFreeze(ctx);
        auto gamma_prod = add_elementwise(
            ctx, nvinfer1::ElementWiseOperation::kPROD, normalized, gamma, (util::node_info(n) + "_gamma").c_str());
        normalized = gamma_prod->getOutput(0);
      }

      // beta
      if (args[3].IValue()->isTensor()) {
        auto beta = args[3].ITensorOrFreeze(ctx);
        auto beta_sum = add_elementwise(
            ctx, nvinfer1::ElementWiseOperation::kSUM, normalized, beta, (util::node_info(n) + "_beta").c_str());
        normalized = beta_sum->getOutput(0);
      }

      ctx->AssociateValueAndTensor(n->outputs()[0], normalized);
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
