#include <torch/torch.h>
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

#if NV_TENSORRT_MAJOR > 7
// clang-format off
auto quantization_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This aten operator is generated from torch.fake_quantize_per_tensor_affine op in Pytorch python API.
              // Example usage: https://github.com/pytorch/pytorch/blob/3139722679a9813ac8e60a07e577cd85c4b06a84/torch/quantization/fake_quantize.py#L145
              auto input = args[0].ITensorOrFreeze(ctx);
              auto scale = args[1].unwrapToScalar().to<float>();
              auto scaleTensor = tensor_to_const(ctx, torch::tensor({scale}));
              // Add and configure a QuantizeLayer.
              nvinfer1::IQuantizeLayer* quantize_layer = ctx->net->addQuantize(*input, *scaleTensor);
              quantize_layer->setAxis(0);

              // Add and configure DequantizeLayer following a QuantizeLayer
              nvinfer1::IDequantizeLayer* dequantize_layer = ctx->net->addDequantize(*quantize_layer->getOutput(0), *scaleTensor);
              dequantize_layer->setAxis(0);

              auto qdq_out = ctx->AssociateValueAndTensor(n->outputs()[0], dequantize_layer->getOutput(0));
              LOG_DEBUG("[fake_quantize_per_tensor_affine] Output tensor shape: " << qdq_out->getDimensions());

              return true;
            }})
  .pattern({"aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This aten operator is generated from torch.fake_quantize_per_channel_affine op in Pytorch python API.
              // Example usage: https://github.com/pytorch/pytorch/blob/3139722679a9813ac8e60a07e577cd85c4b06a84/torch/quantization/fake_quantize.py#L145
              auto input = args[0].ITensorOrFreeze(ctx);
              auto scale = args[1].ITensorOrFreeze(ctx);
              int64_t axis = args[3].unwrapToScalar().to<int64_t>();
              // Add and configure a QuantizeLayer.
              nvinfer1::IQuantizeLayer* quantize_layer = ctx->net->addQuantize(*input, *scale);
              // Set a channel axis which represents output channels
              quantize_layer->setAxis(axis);

              // Add and configure a DequantizeLayer.
              nvinfer1::IDequantizeLayer* dequantize_layer = ctx->net->addDequantize(*quantize_layer->getOutput(0), *scale);
              dequantize_layer->setAxis(axis);
              auto qdq_out = ctx->AssociateValueAndTensor(n->outputs()[0], dequantize_layer->getOutput(0));

              LOG_DEBUG("[fake_quantize_per_channel_affine] Ouput tensor shape: " << qdq_out->getDimensions());

              return true;
            }});
// clang-format on
#endif
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
