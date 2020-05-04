#include "core/conversion/converters/converters.h"

#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

static auto shape_registrations = RegisterNodeConversionPatterns()
  .pattern({
    // To use in static input size cases (explicit batch)
    "aten::size.int(Tensor self, int dim) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor();
      auto in_shape = util::toVec(in->getDimensions());

      auto size = in_shape[args[1].unwrapToInt()];

      ctx->AssociateValueAndIValue(n->outputs()[0], size);
      LOG_DEBUG("Output Value: " << size);
      return true;
    }
  });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
