#include "torch/torch.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"
#include "NvInfer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <vector>

#include <csignal>

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
            auto in = args[0].ITensor();
            auto axis  = args[1].unwrapToInt();
            auto ind = (int32_t) args[2].unwrapToInt();

            // index to access needs to be an at::Tensor
            at::Tensor indices = torch::tensor({ind}).to(torch::kI32);
            auto weights = Weights(ctx, indices);

            // IConstantLayer to convert indices from Weights to ITensor
            auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
            TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
            auto const_out = const_layer->getOutput(0);
            
            // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
            auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
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