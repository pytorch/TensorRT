#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto constant_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"trt::const(Tensor self) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This converter may be abusing what the registry is supposed to be
              // used for Fundimentally this is because of the differing
              // philosophies between TensorRT and PyTorch, i.e. Variables contain
              // Tensors vs. just Tensors
              nvinfer1::ITensor* output;
              if (args[0].isITensor()){
                output = ctx->AssociateValueAndTensor(n->outputs()[0], args[0].ITensor());
              } else{
                auto t = args[0].unwrapToTensor();
                auto const_out = tensor_to_const(ctx, t, util::node_info(n).c_str());
                output = ctx->AssociateValueAndTensor(n->outputs()[0], const_out);
              }
              LOG_DEBUG("Output tensor shape: " << output->getDimensions());

              return true;
            }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
