#include <torch/torch.h>
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto constant_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"trt::const(Tensor self) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This converter may be abusing what the registry is supposed to be
              // used for Fundimentally this is because of the differing
              // philosophies between TensorRT and PyTorch, i.e. Variables contain
              // Tensors vs. just Tensors

              auto t = args[0].unwrapToTensor();
              auto t_weights = Weights(ctx, t);
              auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
              const_layer->setName(util::node_info(n).c_str());
              auto const_out = ctx->AssociateValueAndTensor(n->outputs()[0], const_layer->getOutput(0));

              LOG_DEBUG("Output tensor shape: " << const_out->getDimensions());

              return true;
            }})
  .pattern({"aten::full(int[] size, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto size = args[0].unwrapToIntList();
              auto scalar = args[1].unwrapToScalar().to<float>();
              auto scalar_tensor = torch::full({5}, scalar);
              auto full_tensor = tensor_to_const(ctx, scalar_tensor);
              auto output = ctx->AssociateValueAndTensor(n->outputs()[0], full_tensor);

              LOG_DEBUG("Output tensor shape: " << output->getDimensions());

              return true;
            }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
