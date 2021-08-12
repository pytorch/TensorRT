#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto squeeze_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::squeeze.dim(Tensor(a) self, int dim) -> (Tensor(a))",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto self = args[0].ITensorOrFreeze(ctx);
       auto dim = args[1].unwrapToInt();

       auto selfDim = util::toVec(self->getDimensions());
       if (dim < 0) {
         dim = selfDim.size() + dim;
       }

       if (selfDim[dim] != 1) {
         auto out = ctx->AssociateValueAndTensor(n->outputs()[0], self);

         LOG_DEBUG("Output tensor shape: " << out->getDimensions());

         return true;
       }

       auto shuffle_layer = ctx->net->addShuffle(*self);
       TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
       shuffle_layer->setReshapeDimensions(util::squeezeDims(self->getDimensions(), dim));

       auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_layer->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << out->getDimensions());

       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch