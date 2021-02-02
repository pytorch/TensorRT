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

auto topk_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto self = args[0].ITensorOrFreeze(ctx);
       auto k = args[1].unwrapToInt();
       auto dim = args[2].unwrapToInt();
       auto largest = args[3].unwrapToBool();
       LOG_DEBUG(
           "Note: sorted argument is not used in TensorRT for aten::topk, results will depend on the value of largest");
       // auto sorted = args[4].unwrapToBool(); # Currently unused

       auto selfDim = util::toVec(self->getDimensions());

       // reduceAxes	The reduction dimensions. The bit in position i of bitmask reduceAxes corresponds to explicit
       // dimension i of the result. E.g., the least significant bit corresponds to the first explicit dimension and the
       // next to least significant bit corresponds to the second explicit dimension.

       if (dim < 0) {
         dim = selfDim.size() + dim;
       }

       uint32_t shiftDim = 1 << dim;

       LOG_DEBUG("Output topk reduce dim: " << dim);

       auto TopKOperation = largest ? (nvinfer1::TopKOperation::kMAX) : (nvinfer1::TopKOperation::kMIN);

       auto new_layer = ctx->net->addTopK(*self, TopKOperation, k, shiftDim);

       TRTORCH_CHECK(new_layer, "Unable to create topk layer from node: " << *n);

       auto out0 = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));
       auto out1 = ctx->AssociateValueAndTensor(n->outputs()[1], new_layer->getOutput(1));

       LOG_DEBUG("Output tensor(0) shape: " << out0->getDimensions());
       LOG_DEBUG("Output tensor(1) shape: " << out1->getDimensions());

       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch