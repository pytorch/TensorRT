#include "torch/torch.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto interpolate_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in = args[0].ITensor();

            auto shape = util::toVec(in->getDimensions());

            LOG_DEBUG("Shape of input is" << in);

            std::cout << "TEST!" << std::endl;

            return true;
        }
    });


} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
