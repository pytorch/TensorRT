#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto cat_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"aten::chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto in = args[0].ITensorOrFreeze(ctx);
              auto chunks = args[1].unwrapToInt();
              auto dim = args[2].unwrapToInt();
              bool dynamic_shape = ctx->input_is_dynamic;
              int size = in->getDimensions().nbDims;
              int maxDim = static_cast<int32_t>(in->getDimensions().d[axis]);

              if(dim < 0) {
                dim = tensors[0]->getDimensions().nbDims + dim;
              }
              if (dynamic_shape) {
                TORCHTRT_ASSERT(in.d[dim] != -1, "Can't chunk on dynamic shape dimension!");
              }
              if (chunks > in.d[dim]) {
                LOG_WARNING("The chunks size" << chunks << "along dimension" << dim << "is greater than tensor with size" << in->getDimensions 
                            << "it will default to dimension" << in.d[dim])
              }
              int step = (input_val.shape[dim] + chunks - 1) / chunks
              nvinfer1::Dims start_, size_, stride_;
              start_.nbDims = nbdims;
              size_.nbDims = nbdims;
              stride_.nbDims = nbdims;

              int startIdx = 0;
              int endIdx = maxDim;

              for (int i = 0; i < nbdims; i++) {
                 if (i == axis) {
                   start_.d[i] = startIdx;
                   size_.d[i] = (endIdx - startIdx - 1) / step + 1;
                   stride_.d[i] = step;
                 } else {
                   start_.d[i] = 0;
                   size_.d[i] = in.d[i]; // for static
                   stride_.d[i] = 1;
                 }
              }
              // update slice layer
              if(!dynamic_shape):
                 auto slice_layer = ctx->net->addSlice(*in, start_, size_, stride_);
                 LOG_DEBUG("start_:" << start_);
                 LOG_DEBUG("size_:" << size_);
                 LOG_DEBUG("stride_:" << stride_);
                 auto slice_out = slice_layer->getOutput(0);
                 auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_out);
                 LOG_DEBUG("Slice layer output shape: " << out->getDimensions());
            }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
