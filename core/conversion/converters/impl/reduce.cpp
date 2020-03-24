#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {
auto reduced_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in_tensor = args[0].ITensor();
            auto dim = args[1].unwrapToIntList();
            auto keepdim = args[2].unwrapToBool();

            uint32_t axis_mask = 1 << dim[0];

            LOG_WARNING("Mean converter disregards dtype");
            auto mean_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kAVG, axis_mask, keepdim);
            mean_layer->setName(util::node_info(n).c_str());

            auto out_value = n->outputs()[0];
            auto out_tensor = mean_layer->getOutput(0);
            out_tensor->setName(out_value->debugName().c_str());
            ctx->value_tensor_map[out_value] = out_tensor;
            
            return true;
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch 

