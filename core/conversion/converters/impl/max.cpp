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
auto max_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto self = args[0].ITensorOrFreeze(ctx);
       auto dim = args[1].unwrapToInt();
       auto keep_dims = args[2].unwrapToBool();
       auto selfDim = util::toVec(self->getDimensions());
       if (dim < 0) {
         dim = selfDim.size() + dim;
       }
       uint32_t shiftDim = 1 << dim;
       auto TopKOperation = nvinfer1::TopKOperation::kMAX;
       auto topk_layer = ctx->net->addTopK(*self, TopKOperation, 1, shiftDim);
       TORCHTRT_CHECK(topk_layer, "Unable to create max layer from node: " << *n);
       auto topk_dims = util::toVec(topk_layer->getOutput(0)->getDimensions());

       nvinfer1::ITensor* out0 = nullptr;
       nvinfer1::ITensor* out1 = nullptr;
       if (!keep_dims) {
         if (topk_dims[dim] == 1) {
           auto squeeze_layer = ctx->net->addShuffle(*topk_layer->getOutput(0));
           squeeze_layer->setReshapeDimensions(util::squeezeDims(topk_layer->getOutput(0)->getDimensions(), dim));
           TORCHTRT_CHECK(squeeze_layer, "Unable to create squeeze_layer layer from node: " << *n);
           out0 = ctx->AssociateValueAndTensor(n->outputs()[0], squeeze_layer->getOutput(0));

           auto squeeze_layer_indices = ctx->net->addShuffle(*topk_layer->getOutput(1));
           squeeze_layer_indices->setReshapeDimensions(
               util::squeezeDims(topk_layer->getOutput(1)->getDimensions(), dim));
           TORCHTRT_CHECK(squeeze_layer_indices, "Unable to create squeeze_layer_indices layer from node: " << *n);
           out1 = ctx->AssociateValueAndTensor(n->outputs()[1], squeeze_layer_indices->getOutput(0));
         }
       } else {
         out0 = ctx->AssociateValueAndTensor(n->outputs()[0], topk_layer->getOutput(0));
         out1 = ctx->AssociateValueAndTensor(n->outputs()[1], topk_layer->getOutput(1));
       }

       LOG_DEBUG("Output tensor(0) shape: " << out0->getDimensions());
       LOG_DEBUG("Output tensor(1) shape: " << out1->getDimensions());

       return true;
     }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
