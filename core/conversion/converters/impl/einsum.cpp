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
    {"aten::einsum(str equation, Tensor[] tensors) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       // Extract equation and list of tensors
       auto equation = args[0].unwrapToString();
       auto in = args[1].IValue()->toListRef();

       std::vector<nvinfer1::ITensor*> tensors;

       // Populate vector of ITensor pointers
       for (auto t : in) {
         nvinfer1::ITensor* itensor;

         // Tensor is either an ITensor (wrapped) or PyTorch Tensor
         if (t.isTensor()) {
           auto weight = Weights(ctx, t.toTensor());

           auto const_layer = ctx->net->addConstant(weight.shape, weight.data);
           TORCHTRT_CHECK(const_layer, "Unable to create constant layer from node: " << *n);

           itensor = const_layer->getOutput(0);
         } else {
           auto cont = t.toCustomClass<TensorContainer>();
           itensor = cont->tensor();
         }

         tensors.push_back(itensor);
       }

       // Add Tensor-RT Einsum layer
       auto einsum_layer = ctx->net->addEinsum(tensors.data(), tensors.size(), equation.c_str());
       TORCHTRT_CHECK(einsum_layer, "Unable to create einsum layer from node: " << *n);

       einsum_layer->setName(util::node_info(n).c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], einsum_layer->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
