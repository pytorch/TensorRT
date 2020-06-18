#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {
auto cat_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
    .pattern({
        "aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto ts = args[0].IValue()->toListRef();
            auto dim = args[1].unwrapToInt();

            std::vector<nvinfer1::ITensor*> tensors;
            for (auto t : ts) {
                if (t.isTensor()) {
                    auto torch_tensor = t.toTensor();
                    auto t_weights = Weights(ctx, torch_tensor);
                    auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
                    tensors.push_back(const_layer->getOutput(0));
                } else {
                    auto cont = t.toCustomClass<TensorContainer>();
                    tensors.push_back(cont->tensor());
                }
            }

            auto cat_layer = ctx->net->addConcatenation(tensors.data(), tensors.size());
            cat_layer->setAxis(static_cast<int>(dim));
            auto cat_out = ctx->AssociateValueAndTensor(n->outputs()[0], cat_layer->getOutput(0));

            LOG_DEBUG("Output tensor shape: " << cat_out->getDimensions());

            return true;
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

