#include "torch/torch.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"
#include "NvInfer.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto select_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
    .pattern({
        "aten::select.int(Tensor(a) self, int dim, int index) -> (Tensor(a))",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in = args[0].ITensorOrFreeze(ctx, n);
            auto axis  = args[1].unwrapToInt();

            nvinfer1::ITensor* ind_tensor;

            if (args[2].isITensor()) {
                ind_tensor = args[2].ITensorOrFreeze(ctx, n);
            } else {
                auto weights = Weights(ctx, (int32_t) args[2].unwrapToInt());

                auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);

                ind_tensor = const_layer->getOutput(0);
            }

            // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
            auto gather_layer = ctx->net->addGather(*in, *ind_tensor, axis);
            TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
            auto gather_out = gather_layer->getOutput(0);

            // IShuffleLayer removes redundant dimensions
            auto shuffle_layer = ctx->net->addShuffle(*gather_out);
            TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
            shuffle_layer->setReshapeDimensions(util::unpadDims(gather_out->getDimensions()));
            shuffle_layer->setName(util::node_info(n).c_str());
            auto shuffle_out = shuffle_layer->getOutput(0);

            auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_out);

            LOG_DEBUG("Output tensor shape: " << out->getDimensions());

            return true;
        }
    });

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch