#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
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
    TRTORCH_CHECK(new_layer, "Unable to create " #act " layer from node: " << *n);                                   \
                                                                                                                     \
    new_layer->setName(util::node_info(n).c_str());                                                                  \
    ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));                                          \
    LOG_DEBUG("Output tensor shape: " << new_layer->getOutput(0)->getDimensions());                                  \
                                                                                                                     \
    return true;                                                                                                     \
  }                                                                                                                  \
                                                                                                                     \
  auto act##_registrations TRTORCH_UNUSED =                                                                          \
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

auto acthardtanh TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    auto min = args[1].unwrapToDouble();
                    auto max = args[2].unwrapToDouble();

                    auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kCLIP);
                    TRTORCH_CHECK(new_layer, "Unable to create layer for aten::hardtanh");

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
                    TRTORCH_CHECK(new_layer, "Unable to create layer for aten::hardtanh");

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
                   TRTORCH_CHECK(in_shuffle, "Unable to create resize layer for aten::prelu input");
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
                 TRTORCH_CHECK(out_shuffle, "Unable to create resize layer for aten::prelu output");
                 out_shuffle->setReshapeDimensions(original_shape);
                 out_shuffle->setName(
                     (std::string("[Reshape back to ") + util::toStr(original_shape) + std::string("]")).c_str());
                 out_tensor = out_shuffle->getOutput(0);
               }

               out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern({"aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> (Tensor)",
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
        .pattern({"aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)",
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
        .pattern({"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    auto alpha = args[1].unwrapToDouble();

                    auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kELU);
                    TRTORCH_CHECK(new_layer, "Unable to create layer for aten::elu");
                    new_layer->setAlpha(alpha);

                    new_layer->setName(util::node_info(n).c_str());

                    auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));
                    LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
                    return true;
                  }})
        .pattern({"aten::gelu(Tensor self) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    nvinfer1::DataType type = in->getType();
                    TRTORCH_CHECK(
                        type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF,
                        "gelu only supports kFLOAT and kHALF");
                    std::string pluginName = "CustomGeluPluginDynamic";
                    nvinfer1::PluginFieldCollection fc;
                    std::vector<nvinfer1::PluginField> f;
                    // REVIEW is this right?
                    int type_id = ctx->settings.enabled_precisions.find(nvinfer1::DataType::kHALF) ==
                            ctx->settings.enabled_precisions.end()
                        ? 0
                        : 1; // Integer encoding the DataType (0: FP32, 1: FP16)
                    f.emplace_back(nvinfer1::PluginField("type_id", &type_id, nvinfer1::PluginFieldType::kINT32, 1));
                    fc.nbFields = f.size();
                    fc.fields = f.data();

                    auto creator = getPluginRegistry()->getPluginCreator("CustomGeluPluginDynamic", "1", "");
                    auto gelu_plugin = creator->createPlugin("gelu", &fc);

                    TRTORCH_CHECK(gelu_plugin, "Unable to create gelu plugin from TensorRT plugin registry" << *n);
                    auto new_layer =
                        ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *gelu_plugin);
                    new_layer->setName(util::node_info(n).c_str());
                    auto out_tensor = new_layer->getOutput(0);
                    out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
                    LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
                    return true;
                  }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
