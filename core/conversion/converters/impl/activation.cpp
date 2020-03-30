#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

#define convert(act, trt_type)                                                 \
  bool act(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {        \
    auto in = args[0].ITensor();                                               \
                                                                               \
    auto new_layer =                                                           \
        ctx->net->addActivation(*in, nvinfer1::ActivationType::trt_type);      \
    TRTORCH_CHECK(new_layer,                                                   \
                  "Unable to create " #act " layer from node: " << *n);        \
                                                                               \
    new_layer->setName(util::node_info(n).c_str());                            \
    ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));    \
    LOG_DEBUG("Output tensor shape: "                                          \
               << new_layer->getOutput(0)->getDimensions());                   \
                                                                               \
    return true;                                                               \
  }                                                                            \
                                                                               \
  auto act##_registrations TRTORCH_UNUSED =                                    \
      RegisterNodeConversionPatterns()                                         \
          .pattern({"aten::" #act "(Tensor input) -> (Tensor)",                \
                    [](ConversionCtx* ctx, const torch::jit::Node* n,          \
                       args& args) -> bool { return act(ctx, n, args); }})     \
          .pattern({"aten::" #act "_(Tensor(a!) self) -> (Tensor(a!))",        \
                    [](ConversionCtx* ctx, const torch::jit::Node* n,          \
                       args& args) -> bool { return act(ctx, n, args); }});

//TODO: remove support for conversion of implace operators and move to the functionalization pass

convert(relu, kRELU);
convert(sigmoid, kSIGMOID);
convert(tanh, kTANH);

#undef convert

auto acthardtanh TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({
    "aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
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
    }
  }).pattern({
    //TODO: Remove after functionalization
    "aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> (Tensor(a!))",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
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
    }
  });



} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
