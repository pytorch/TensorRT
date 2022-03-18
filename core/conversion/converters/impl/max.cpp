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
             auto k = 1;
             auto dim = args[1].unwrapToInt();
             auto largest = true;
             auto selfDim = util::toVec(self->getDimensions());
             if (dim < 0) {
                 dim = selfDim.size() + dim;
             }
             uint32_t shiftDim = 1 << dim;

             auto TopKOperation = largest ? (nvinfer1::TopKOperation::kMAX) : (nvinfer1::TopKOperation::kMIN);

             auto new_layer = ctx->net->addTopK(*self, TopKOperation, 1, shiftDim);
             TORCHTRT_CHECK(new_layer, "Unable to create max layer from node: " << *n);

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
} // namespace torch_tensorrt
