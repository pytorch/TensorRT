#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool relu(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {
     auto in = args[0].ITensor();

     auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kRELU);
     TRTORCH_CHECK(new_layer, "Unable to create ReLU layer from node: " << *n);

     new_layer->setName(util::node_info(n).c_str());
     auto out_value = n->outputs()[0];
     auto out_tensor = new_layer->getOutput(0);
     out_tensor->setName(out_value->debugName().c_str());
     ctx->value_tensor_map[out_value] = out_tensor;
     LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

     return true;
}

auto relu_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::relu(Tensor input) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            return relu(ctx, n, args);
        }
    }).pattern({
        "aten::relu_(Tensor(a!) self) -> (Tensor(a!))",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            return relu(ctx, n, args);
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // trtorch 
