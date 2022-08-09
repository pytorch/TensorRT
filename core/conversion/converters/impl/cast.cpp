#include <torch/torch.h>
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto cast_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::to.dtype(Tensor self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto output_dtype = args[1].unwrapToScalar().to<int64_t>();
               auto scalar_dtype = static_cast<at::ScalarType>(output_dtype);
               nvinfer1::DataType trt_dtype;
               if (scalar_dtype == at::kLong) {
                 LOG_WARNING("Truncating aten::to output type from at::kLong to at::kInt");
                 trt_dtype = nvinfer1::DataType::kINT32;
               } else {
                 trt_dtype = util::ScalarTypeToTRTDataType(static_cast<at::ScalarType>(output_dtype));
               }
               auto casted_itensor = castITensor(ctx, self, trt_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.dtype] Output tensor shape: " << output->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::to.device(Tensor(a) self, Device device, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // what this function does is basically the same with the previous one, however, we cannot lower this
               // signature to previous one because this will incur the device issues when we run Torchscript module in
               // later shape analysis phase of fallback
               auto self = args[0].ITensorOrFreeze(ctx);
               auto output_dtype = args[2].unwrapToScalar().to<int64_t>();
               auto scalar_dtype = static_cast<at::ScalarType>(output_dtype);
               nvinfer1::DataType trt_dtype;
               if (scalar_dtype == at::kLong) {
                 LOG_WARNING("Truncating aten::to output type from at::kLong to at::kInt");
                 trt_dtype = nvinfer1::DataType::kINT32;
               } else {
                 trt_dtype = util::ScalarTypeToTRTDataType(static_cast<at::ScalarType>(output_dtype));
               }
               auto casted_itensor = castITensor(ctx, self, trt_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.device] Output tensor shape: " << output->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               nvinfer1::DataType other_dtype = args[1].ITensorOrFreeze(ctx)->getType();
               auto casted_itensor = castITensor(ctx, self, other_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.other] Output tensor shape: " << output->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               if (args[2].isIValue() && !args[2].IValue()->isScalar()) {
                 auto output = ctx->AssociateValueAndTensor(n->outputs()[0], self);
                 LOG_DEBUG("[aten::to.prim_Device] Output tensor shape: " << output->getDimensions());
                 return true;
               }

               auto output_dtype = args[2].unwrapToScalar().to<int64_t>();
               auto trt_dtype = util::ScalarTypeToTRTDataType(static_cast<at::ScalarType>(output_dtype));
               auto casted_itensor = castITensor(ctx, self, trt_dtype);
               auto output = ctx->AssociateValueAndTensor(n->outputs()[0], casted_itensor);
               LOG_DEBUG("[aten::to.prim_Device] Output tensor shape: " << output->getDimensions());

               return true;
             }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
