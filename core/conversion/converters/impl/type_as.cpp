#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto type_as_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::type_as(Tensor self, Tensor other) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto targetTensor = args[1].ITensor();

       auto identity = ctx->net->addIdentity(*in);
       TRTORCH_CHECK(identity, "Unable to create layer for aten::type_as");
       identity->setOutputType(0, targetTensor->getType());

       identity->setName(util::node_info(n).c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], identity->getOutput(0));

       LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
