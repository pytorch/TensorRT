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
              int maxDim = static_cast<int32_t>(in->getDimensions().d[dim]);

              c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
              c10::TypePtr elementType = lt->getElementType();

              int offset = 0;
              if(dim < 0) {
                dim = in->getDimensions().nbDims + dim;
              }
              if (dynamic_shape) {
                TORCHTRT_ASSERT(in->getDimensions().d[dim] != -1, "Can't chunk on dynamic shape dimension!");
              }
              if (chunks > in->getDimensions().d[dim]) {
                LOG_WARNING("The chunks size" << chunks << "along dimension" << dim << "is greater than tensor with size" << in->getDimensions().d[dim]
                            << "it will default to dimension" << in->getDimensions().d[dim]);
              }
              int step = (maxDim + chunks - 1) / chunks;
              nvinfer1::Dims start_, size_, stride_;
              int nbdims = in->getDimensions().nbDims;
              start_.nbDims = nbdims;
              size_.nbDims = nbdims;
              stride_.nbDims = nbdims;

              for (int i = 0; i < nbdims; i++) {
                start_.d[i] = 0;
                size_.d[i] = 0;
                stride_.d[i] = 1;
              }
              // update slice layer
              auto list = c10::impl::GenericList(elementType);
              list.reserve(chunks);
              if(!dynamic_shape) {
                for (int chunk = 0; chunk < chunks; chunk++) {
                  for (int i = 0; i < nbdims; i++) {
                    if (i == dim) {
                      start_.d[i] = offset;
                      size_.d[i] = std::min(step, maxDim - offset);
                    }
                  }
                  LOG_DEBUG("start_:" << start_);
                  LOG_DEBUG("size_:" << size_);
                  LOG_DEBUG("stride_:" << stride_);
                  auto slice_layer = ctx->net->addSlice(*in, start_, size_, stride_);
                  auto tensor_holder = TensorContainer();
                  tensor_holder.hold_tensor(slice_layer->getOutput(0));
                  auto ival = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
                  list.emplace_back(ival);
                  offset = offset + step;
                }
              }
              auto split_output_ivalue = std::move(torch::jit::IValue(list));
              ctx->AssociateValueAndIValue(n->outputs()[0], split_output_ivalue);
              return true;
            }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
