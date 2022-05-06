#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

#include <torch/torch.h>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {


auto bitwisenot TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::bitwise_not(Tensor self) -> Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    nvinfer1::ILayer* out;
                    
                    if(in->getType() == nvinfer1::DataType::kINT32) {
                      // Integer case
                      auto one = torch::tensor({1}, util::TRTDataTypeToScalarType(in->getType()));
                      auto one_const = tensor_to_const(ctx, one);
                      auto neg = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kNEG);
                      TORCHTRT_CHECK(neg, "Unable to create neg unary layer from node: " << *n);
                      out = add_elementwise(
                          ctx, nvinfer1::ElementWiseOperation::kSUB, neg->getOutput(0),
                          one_const, util::node_info(n));
                      TORCHTRT_CHECK(out, "Unable to create sub layer from node: " << *n);
                    } else if(in->getType() == nvinfer1::DataType::kBOOL) {
                      // Boolean case
                      out = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kNOT);
                      TORCHTRT_CHECK(out, "Unable to create logical not layer from node: " << *n);
                    } else {
                      LOG_ERROR("Input tensor must be 32 bit integer or boolean");
                      return false;
                    }

                    out->setName(util::node_info(n).c_str());
                    auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0],
                                                                   out->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

                    return true;
                  }});


} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
