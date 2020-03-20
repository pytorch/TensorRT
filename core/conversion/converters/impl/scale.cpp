#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto scale_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::sqrt(Tensor self) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto self = args[0].ITensor();

            auto shape = util::toVec(self->getDimensions());
            if (shape.size() < 4) {
                 auto new_shape = util::toDimsPad(shape, 4);
                 LOG_DEBUG("Input shape is less than 4D got: " << util::toDims(shape) << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
                 auto shuffle = ctx->net->addShuffle(*self);
                 shuffle->setReshapeDimensions(new_shape);
                 shuffle->setName(std::string("[Reshape self to " + util::toStr(new_shape) + ']').c_str());
                 self = shuffle->getOutput(0);
             }
            
            auto half = Weights(ctx, 0.5);
            auto sqrt = ctx->net->addScale(*self, nvinfer1::ScaleMode::kUNIFORM, Weights().data, Weights().data, half.data);
            sqrt->setName(util::node_info(n).c_str());
            auto out_value = n->outputs()[0];
            auto out_tensor = sqrt->getOutput(0);
            
            // if (shape.size() < 4) {
            //     LOG_DEBUG("Input shape was less than 4D and was reshaped, inserting shuffle layer to reshape back to original shape: " << util::toDims(shape));
            //      auto shuffle = ctx->net->addShuffle(*self);
            //      shuffle->setReshapeDimensions(util::toDims(shape));
            //      shuffle->setName(std::string("[Reshape self to " + util::toStr(util::toDims(shape)) + ']').c_str());
            //      out_tensor = shuffle->getOutput(0);
            //  }

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
} // trtorch 


























