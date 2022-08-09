#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto stack_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::stack(Tensor[] tensors, int dim=0) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].IValue()->toListRef();
       auto dim = args[1].unwrapToInt();
       if (-1 == dim) {
         auto first_in = in[0];
         if (first_in.isTensor()) {
           dim = first_in.toTensor().ndimension();
         } else {
           dim = first_in.toCustomClass<TensorContainer>()->tensor()->getDimensions().nbDims;
         }
       }

       std::vector<nvinfer1::ITensor*> tensors;
       for (auto t : in) {
         nvinfer1::ITensor* itensor;

         if (t.isTensor()) {
           auto weight = Weights(ctx, t.toTensor());

           auto const_layer = ctx->net->addConstant(weight.shape, weight.data);
           TORCHTRT_CHECK(const_layer, "Unable to create constant layer from node: " << *n);

           itensor = const_layer->getOutput(0);
         } else {
           auto cont = t.toCustomClass<TensorContainer>();
           itensor = cont->tensor();
         }

         auto shuffle_layer = ctx->net->addShuffle(*itensor);
         TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
         shuffle_layer->setReshapeDimensions(util::unsqueezeDims(itensor->getDimensions(), dim));

         tensors.push_back(shuffle_layer->getOutput(0));
       }

       auto concat_layer = ctx->net->addConcatenation(tensors.data(), tensors.size());
       TORCHTRT_CHECK(concat_layer, "Unable to create concatenation layer from node: " << *n);
       concat_layer->setAxis(static_cast<int>(dim));
       auto out = ctx->AssociateValueAndTensor(n->outputs()[0], concat_layer->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << out->getDimensions());

       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
