#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto unsqueeze_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::unsqueeze(Tensor(a) self, int dim) -> (Tensor(a))",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto self = args[0].ITensorOrFreeze(ctx);
       auto dim = args[1].unwrapToInt();

       auto selfDim = util::toVec(self->getDimensions());
       int64_t nbDims = selfDim.size();
       TORCHTRT_CHECK(
           dim <= nbDims && dim >= -(nbDims + 1),
           "Dimension out of range (expected to be in range of [" << -(nbDims + 1) << ", " << nbDims << "], but got "
                                                                  << dim << ")");
       if (dim < 0) {
         dim = nbDims + 1 + dim;
       }

       auto shuffle_layer = ctx->net->addShuffle(*self);
       TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
       shuffle_layer->setReshapeDimensions(util::unsqueezeDims(self->getDimensions(), dim));

       auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_layer->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << out->getDimensions());

       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
