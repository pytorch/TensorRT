#include <torch/torch.h>
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool register_cast_layer(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* input,
    nvinfer1::DataType output_dtype) {
  auto cast_layer = ctx->net->addIdentity(*input);
  cast_layer->setName(util::node_info(n).c_str());
  cast_layer->setOutputType(0, output_dtype);

  auto output = ctx->AssociateValueAndTensor(n->outputs()[0], cast_layer->getOutput(0));
  LOG_DEBUG("Cast layer output tensor shape: " << output->getDimensions());

  return true;
}

auto cast_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::to.dtype(Tensor self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto output_dtype = args[1].unwrapToScalar().to<int64_t>();
               auto aten_to_trt_dtype_map = util::get_aten_trt_type_map();
               TRTORCH_CHECK(
                   aten_to_trt_dtype_map.find(static_cast<at::ScalarType>(output_dtype)) != aten_to_trt_dtype_map.end(),
                   "Conversion to desired datatype is not supported");
               return register_cast_layer(
                   ctx, n, self, aten_to_trt_dtype_map.at(static_cast<at::ScalarType>(output_dtype)));
             }})
        .pattern(
            {"aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               nvinfer1::DataType other_dtype = args[1].ITensorOrFreeze(ctx)->getType();
               auto aten_to_trt_dtype_map = util::get_aten_trt_type_map();
               bool is_datatype_supported = false;
               for (auto it = aten_to_trt_dtype_map.begin(); it != aten_to_trt_dtype_map.end(); ++it) {
                 if (it->second == other_dtype) {
                   is_datatype_supported = true;
                 }
               }
               TRTORCH_CHECK(is_datatype_supported, "Conversion to desired datatype is not supported");
               return register_cast_layer(ctx, n, self, other_dtype);
             }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
