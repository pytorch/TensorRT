#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace my_custom_converters {

auto actelu = torch_tensorrt::core::conversion::converters::RegisterNodeConversionPatterns().pattern(
    {"aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> (Tensor)",
     [](torch_tensorrt::core::conversion::ConversionCtx* ctx,
        const torch::jit::Node* n,
        torch_tensorrt::core::conversion::converters::args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       auto alpha = args[1].unwrapToDouble();

       auto new_layer = ctx->net->addActivation(*in, nvinfer1::ActivationType::kELU);
       if (!(new_layer)) {
         std::cerr << "Unable to create layer for aten::elu" << std::endl;
       }

       new_layer->setAlpha(alpha);
       new_layer->setName(torch_tensorrt::core::util::node_info(n).c_str());
       ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

       return true;
     }});

} // namespace my_custom_converters
