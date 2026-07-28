#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

#define convert(act, trt_type)                                                                                       \
  bool act(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {                                              \
    auto in = args[0].ITensorOrFreeze(ctx);                                                                          \
                                                                                                                     \
    auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::trt_type);                               \
    TORCHTRT_CHECK(new_layer, "Unable to create " #act " layer from node: " << *n);                                  \
                                                                                                                     \
    new_layer->setName(util::node_info(n).c_str());                                                                  \
    ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));                                          \
    LOG_DEBUG("Output tensor shape: " << new_layer->getOutput(0)->getDimensions());                                  \
                                                                                                                     \
    return true;                                                                                                     \
  }                                                                                                                  \
                                                                                                                     \
  auto act##_registrations TORCHTRT_UNUSED =                                                                         \
      RegisterNodeConversionPatterns()                                                                               \
          .pattern(                                                                                                  \
              {"aten::" #act "(Tensor input) -> (Tensor)",                                                           \
               [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool { return act(ctx, n, args); }}) \
          .pattern(                                                                                                  \
              {"aten::" #act "_(Tensor(a!) self) -> (Tensor(a!))",                                                   \
               [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool { return act(ctx, n, args); }});

// TODO: remove support for conversion of implace operators and move to the
// functionalization pass

convert(relu, kRELU);
convert(sigmoid, kSIGMOID);
convert(tanh, kTANH);

#undef convert

auto acthardtanh TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto min = args[1].unwrapToDouble();
               auto max = args[2].unwrapToDouble();

               auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kCLIP);
               TORCHTRT_CHECK(new_layer, "Unable to create layer for aten::hardtanh");

               new_layer->setAlpha(min);
               new_layer->setBeta(max);

               new_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern({// TODO: Remove after functionalization
                  "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor(a!))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    auto min = args[1].unwrapToDouble();
                    auto max = args[2].unwrapToDouble();

                    auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kCLIP);
                    TORCHTRT_CHECK(new_layer, "Unable to create layer for aten::hardtanh");

                    new_layer->setAlpha(min);
                    new_layer->setBeta(max);

                    new_layer->setName(util::node_info(n).c_str());
                    auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

                    LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
                    return true;
                  }})
        .pattern(
            {"aten::prelu(Tensor self, Tensor weight) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto slopes = args[1].unwrapToTensor();

               bool to_reshape = false;
               auto original_shape = in->getDimensions();

               // Out_tensor of ParametricReLU shape is all 0, when slopes nDims is not equal to in nDims.
               // Since make sure splopes nDims is equal to in nDims.
               if (slopes.ndimension() == 1 && original_shape.nbDims != slopes.ndimension()) {
                 std::vector<int64_t> slopes_new_shape(original_shape.nbDims, 1);
                 auto first_inputs_allowed_formats = ctx->net->getInput(0)->getAllowedFormats();
                 for (size_t inputs_index = 1; inputs_index < ctx->num_inputs; inputs_index++) {
                   auto inputs_allowed_formats = ctx->net->getInput(inputs_index)->getAllowedFormats();
                   TORCHTRT_CHECK(
                       first_inputs_allowed_formats == inputs_allowed_formats,
                       "Unable to create batch prelu layer from node,since the formats(like NHWC or NCHW) of inputs is different: "
                           << *n);
                 }
                 if (1U << static_cast<int>(nvinfer1::TensorFormat::kLINEAR) == first_inputs_allowed_formats) {
                   slopes_new_shape[1] = slopes.sizes().vec()[0];
                 } else {
                   slopes_new_shape[original_shape.nbDims - 1] = slopes.sizes().vec()[0];
                 }
                 slopes = slopes.reshape(slopes_new_shape);
               }

               if (slopes.numel() != 1 &&
                   !util::broadcastable(
                       in->getDimensions(),
                       util::toDims(slopes.sizes()),
                       /*multidirectional=*/false)) {
                 if (util::volume(in->getDimensions()) == util::volume(util::toDims(slopes.sizes()))) {
                   to_reshape = true;
                   LOG_DEBUG(
                       "Input shape is not broadcastable inserting shuffle layers to reshape to "
                       << util::toDims(slopes.sizes()));
                   auto in_shuffle = ctx->net->addShuffle(*in);
                   TORCHTRT_CHECK(in_shuffle, "Unable to create resize layer for aten::prelu input");
                   in_shuffle->setReshapeDimensions(util::toDims(slopes.sizes()));
                   in_shuffle->setName(
                       std::string("[Reshape in to " + util::toStr(util::toDims(slopes.sizes())) + " for broadcasting]")
                           .c_str());
                   in = in_shuffle->getOutput(0);
                 }
               }

               auto slope_tensor = tensor_to_const(ctx, slopes);
               auto new_layer = ctx->net->addParametricReLU(*in, *slope_tensor);
               new_layer->setName(util::node_info(n).c_str());
               auto out_tensor = new_layer->getOutput(0);

               if (to_reshape) {
                 auto out_shuffle = ctx->net->addShuffle(*out_tensor);
                 TORCHTRT_CHECK(out_shuffle, "Unable to create resize layer for aten::prelu output");
                 out_shuffle->setReshapeDimensions(original_shape);
                 out_shuffle->setName(
                     (std::string("[Reshape back to ") + util::toStr(original_shape) + std::string("]")).c_str());
                 out_tensor = out_shuffle->getOutput(0);
               }

               out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto negative_slopeScalar = args[1].unwrapToScalar().to<float>();

               auto new_layer = ctx->net->addActivation(*self, nvinfer1::ActivationType::kLEAKY_RELU);
               new_layer->setAlpha(negative_slopeScalar);

               new_layer->setName(util::node_info(n).c_str());
               auto out_tensor = new_layer->getOutput(0);
               out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto negative_slopeScalar = args[1].unwrapToScalar().to<float>();

               auto new_layer = ctx->net->addActivation(*self, nvinfer1::ActivationType::kLEAKY_RELU);
               new_layer->setAlpha(negative_slopeScalar);
               new_layer->setName(util::node_info(n).c_str());
               auto out_tensor = new_layer->getOutput(0);
               out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto alpha = args[1].unwrapToDouble();

               auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kELU);
               TORCHTRT_CHECK(new_layer, "Unable to create layer for aten::elu");
               new_layer->setAlpha(alpha);

               new_layer->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
