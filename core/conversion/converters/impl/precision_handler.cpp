#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto constant_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"trtorch::start_setting_precision(Tensor _0, int _1) -> (Tensor _0)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This converter is just a dummy op which indicates that user is planning to set
              // a layer precision (indicated by integer argument) for the following layers in the network until it sees
              // trtorch::stop_setting_precision op.
              auto in = args[0].ITensorOrFreeze(ctx);
              int64_t dtype = args[1].unwrapToInt();
              auto trt_dtype = util::ScalarTypeToTRTDataType(static_cast<c10::ScalarType>(dtype));
              TRTORCH_CHECK(trt_dtype == nvinfer1::DataType::kFLOAT || trt_dtype == nvinfer1::DataType::kHALF ||
                            trt_dtype == nvinfer1::DataType::kINT8,
                            "Unsupported layer precision : " << static_cast<c10::ScalarType>(dtype) << ". Please select among torch.float32|torch.float16|torch.int8.");
              LOG_DEBUG("Explicitly setting all the following layer precisions to " << trt_dtype << " until the node trtorch::stop_setting_precision is found.");
              // If users want to explicitly set layer precision, make sure it is strictly enforced.
              ctx->set_layer_precision = true;
              ctx->next_precision = trt_dtype;
              ctx->settings.strict_types = true;
              ctx->AssociateValueAndTensor(n->outputs()[0], in);

              return true;
            }})
  .pattern({"trtorch::start_setting_precision_with_dr(Tensor _0, int _1, float _2, float _3, float _4, float _5) -> (Tensor _0)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This converter is just a dummy op which indicates that user is planning to set
              // a layer precision (indicated by integer argument) for the following layers in the network until it sees
              // trtorch::stop_setting_precision op.
              auto in = args[0].ITensorOrFreeze(ctx);
              int64_t dtype = args[1].unwrapToInt();
              auto input_min = static_cast<float>(args[2].unwrapToDouble());
              auto input_max = static_cast<float>(args[3].unwrapToDouble());
              auto output_min = static_cast<float>(args[4].unwrapToDouble());
              auto output_max = static_cast<float>(args[5].unwrapToDouble());
              // Set dynamic range for input of this node which will be input to the next layer as well (since this is just an identity operation)
              in->setDynamicRange(input_min, input_max);
              // Set dynamic range for output of the next layer
              ctx->layer_output_dr = std::make_tuple(output_min, output_max);
              auto trt_dtype = util::ScalarTypeToTRTDataType(static_cast<c10::ScalarType>(dtype));
              TRTORCH_CHECK(trt_dtype == nvinfer1::DataType::kFLOAT || trt_dtype == nvinfer1::DataType::kHALF ||
                            trt_dtype == nvinfer1::DataType::kINT8,
                            "Unsupported layer precision : " << static_cast<c10::ScalarType>(dtype) << ". Please select among torch.float32|torch.float16|torch.int8.");
              LOG_DEBUG("Explicitly setting all the following layer precisions to " << trt_dtype << " until the node trtorch::stop_setting_precision is found.");
              // If users want to explicitly set layer precision, make sure it is strictly enforced.
              ctx->set_layer_precision = true;
              ctx->next_precision = trt_dtype;
              ctx->settings.strict_types = true;
              ctx->AssociateValueAndTensor(n->outputs()[0], in);

              return true;
            }})
  .pattern({"trtorch::stop_setting_precision(Tensor _0) -> (Tensor _0)",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This converter is just a dummy op which indicates that explicit layer precision is disabled now.
              // The precision for layers in the network wrapped between trtorch::start_setting_precision and trtorch::stop_setting_precision
              // nodes are explictly set. trtorch::stop_setting_precision signals end of explicit layer precision setting.
              auto in = args[0].ITensorOrFreeze(ctx);
              ctx->next_precision = nvinfer1::DataType::kFLOAT;
              ctx->set_layer_precision = false;
              ctx->AssociateValueAndTensor(n->outputs()[0], in);
              return true;
            }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
