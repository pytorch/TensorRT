#include "torch/torch.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "NvInfer.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {


auto detach_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
        {"aten::detach(Tensor self) -> (Tensor)",
          [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto self = args[0].ITensorOrFreeze(ctx);
            auto identity = ctx->net->addIdentity(*self);
            TRTORCH_CHECK(identity, "Unable to create identity layer from node: " << *n);
            identity->setName(util::node_info(n).c_str());

            auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], identity->getOutput(0));
            LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            return true;
            }});



} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch