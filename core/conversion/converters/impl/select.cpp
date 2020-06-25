#include "torch/torch.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"
#include "NvInfer.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <vector>

#include <csignal>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto select_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
    .pattern({
        "aten::select.int(Tensor(a) self, int dim, int index) -> (Tensor(a))",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            std::cout << "select.int converter recognized" << std::endl;

            auto in = args[0].ITensor();
            auto axis  = args[1].unwrapToInt();
            auto ind = (int32_t) args[2].unwrapToInt();

            // tried: vector for input
            //std::vector<int32_t> indices_input = {ind};

            auto options = torch::TensorOptions().device(torch::kCUDA, 1).dtype(torch::kInt32);
            at::Tensor indices = torch::tensor(torch::detail::TensorDataContainer(ind), options);
            
            auto weights = Weights(ctx, indices);
            // manually setting weights
            // weights.data.type = nvinfer1::DataType::kINT32;

            auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
            const_layer->setName(util::node_info(n).c_str());
            // manually setting output type
            // const_layer->setOutputType(0, nvinfer1::DataType::kINT32);

            auto const_out = ctx->AssociateValueAndTensor(n->outputs()[0], const_layer->getOutput(0)); 
            
            auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
            gather_layer->setName(util::node_info(n).c_str());
            // manually setting output type
            // gather_layer->setOutputType(0, nvinfer1::DataType::kINT32);

            auto gather_output = ctx->AssociateValueAndTensor(n->outputs()[0], gather_layer->getOutput(0));

            LOG_DEBUG("Output tensor shape: " << gather_output->getDimensions());
            
            // for debugging
            // std::raise(SIGTRAP);

            return true;
        }
    });

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch