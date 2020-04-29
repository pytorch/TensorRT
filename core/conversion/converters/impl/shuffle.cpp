#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

static auto shuffle_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::reshape(Tensor self, int[] shape) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in = args[0].ITensor();
            auto new_shape = util::toDimsPad(args[1].unwrapToIntList(), 2);

            auto shuffle = ctx->net->addShuffle(*in);
            TRTORCH_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
            shuffle->setReshapeDimensions(new_shape);
            shuffle->setName(util::node_info(n).c_str());

            auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
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
