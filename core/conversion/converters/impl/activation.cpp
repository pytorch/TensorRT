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
    auto out_value = n->outputs()[0];                                          \
    auto out_tensor = new_layer->getOutput(0);                                 \
    out_tensor->setName(out_value->debugName().c_str());                       \
    ctx->value_tensor_map[out_value] = out_tensor;                             \
    LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());         \
                                                                               \
    return true;                                                               \
  }                                                                            \
                                                                               \
  auto act##_registrations TRTORCH_UNUSED =                                    \
      RegisterNodeConversionPatterns()                                         \
          .pattern({"aten::" #act "(Tensor input) -> (Tensor)",                \
                    [](ConversionCtx *ctx, const torch::jit::Node *n,          \
                       args &args) -> bool { return act(ctx, n, args); }})     \
          .pattern({"aten::" #act "_(Tensor(a!) self) -> (Tensor(a!))",        \
                    [](ConversionCtx *ctx, const torch::jit::Node *n,          \
                       args &args) -> bool { return act(ctx, n, args); }});

convert(relu, kRELU);
convert(sigmoid, kSIGMOID);
convert(tanh, kTANH);

#undef convert
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
