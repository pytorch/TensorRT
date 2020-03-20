#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto unary_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::log(Tensor self) -> Tensor",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in = args[0].ITensor();
            auto log = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kLOG);

            TRTORCH_CHECK(log, "Unable to create log layer from node: " << *n);

            log->setName(util::node_info(n).c_str());
            auto out_value = n->outputs()[0];
            auto out_tensor = log->getOutput(0);
            out_tensor->setName(out_value->debugName().c_str());
            ctx->value_tensor_map[out_value] = out_tensor;
            LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
            return true;
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch 
