#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

/*
 * Helper functions
 */
void create_plugin(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* in,
    int64_t order,
    std::vector<int32_t> axes,
    bool keep_dims,
    const char* name) {
  LOG_WARNING("Normalize layer will be run through ATen, not TensorRT. Performance may be lower than expected");
  nvinfer1::PluginFieldCollection fc;
  std::vector<nvinfer1::PluginField> f;
  f.emplace_back(nvinfer1::PluginField("order", &order, nvinfer1::PluginFieldType::kINT32, 1));
  f.emplace_back(nvinfer1::PluginField("axes", axes.data(), nvinfer1::PluginFieldType::kINT32, axes.size()));
  f.emplace_back(nvinfer1::PluginField("keep_dims", &keep_dims, nvinfer1::PluginFieldType::kINT32, 1));
  fc.nbFields = f.size();
  fc.fields = f.data();

  auto inputnbDims = in->getDimensions().nbDims;
  for (int64_t i = 0; i < (int64_t)axes.size(); i++) {
    if (axes[i] < 0) {
      axes[i] += inputnbDims;
    }
    if (axes[i] > inputnbDims - 1) {
      TORCHTRT_THROW_ERROR("Axis of normalization layer cannot exceed input rank");
    }
  }

  auto creator = getPluginRegistry()->getPluginCreator("NormalizePlugin", "1", "torch_tensorrt");
  auto plugin = creator->createPlugin(name, &fc);
  auto normalize_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
  TORCHTRT_CHECK(normalize_layer, "Unable to create normalization plugin from node" << *n);

  normalize_layer->setName(util::node_info(n).c_str());

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], normalize_layer->getOutput(0));

  LOG_DEBUG("Normalize layer output tensor shape: " << layer_output->getDimensions());
}

int32_t axes_mask_from_axes_values(
    const torch::jit::Node* n,
    int32_t nb_dims,
    const std::vector<int64_t>& axes_values) {
  int32_t axes_mask = 0;
  for (size_t i = 0UL; i < axes_values.size(); ++i) {
    auto axis = axes_values[i];
    if (axis < 0) {
      axis += nb_dims;
    }
    TORCHTRT_CHECK(
        axis < nb_dims, util::node_info(n) << " axis " << i << " with value: " << axis << " exceeds input rank");
    axes_mask += 1 << axis;
  }
  return axes_mask;
}

nvinfer1::ITensor* frobenius_norm(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* self,
    int32_t axes_mask,
    bool keep_dims) {
  auto squared_layer =
      add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, self, util::node_info(n) + "_squared");
  TORCHTRT_CHECK(squared_layer, "Unabled to create square layer from node: " << *n);
  auto squared_output = squared_layer->getOutput(0);

  auto sum_layer = ctx->net->addReduce(*squared_output, nvinfer1::ReduceOperation::kSUM, axes_mask, keep_dims);
  TORCHTRT_CHECK(sum_layer, "Unable to create sum layer from node: " << *n);
  sum_layer->setName((util::node_info(n) + "_sum").c_str());
  auto sum_output = sum_layer->getOutput(0);
  LOG_DEBUG("SUM SHAPE: " << sum_output->getDimensions());

  auto sqrt_layer = ctx->net->addUnary(*sum_output, nvinfer1::UnaryOperation::kSQRT);
  TORCHTRT_CHECK(sqrt_layer, "Unable to create sqrt layer from node: " << *n);
  sqrt_layer->setName((util::node_info(n) + "_sqrt").c_str());
  auto sqrt_output = sqrt_layer->getOutput(0);
  return sqrt_output;
}

auto normalize_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto in_shape = util::toVec(in->getDimensions());
               auto order = args[1].unwrapToScalar().to<int32_t>();
               auto axes_values = args[2].unwrapToIntList().vec();
               std::vector<int32_t> axes(axes_values.begin(), axes_values.end());
               auto keep_dims = (int32_t)args[3].unwrapToBool();
               LOG_DEBUG("Order of normalize_plugin: " << order);
               LOG_DEBUG("Axis: " << axes);
               LOG_DEBUG("keep_dims: " << keep_dims);
               create_plugin(ctx, n, in, order, axes, keep_dims, "NormalizePluginTorchTRT");
               return true;
             }

            })
        .pattern(
            {"aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto axes_values = args[1].unwrapToIntList().vec();
               auto keep_dims = args[2].unwrapToBool();

               auto axes_mask = axes_mask_from_axes_values(n, self->getDimensions().nbDims, axes_values);

               auto norm = frobenius_norm(ctx, n, self, axes_mask, keep_dims);
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], norm);
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, int? dtype=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
               auto self = args[0].ITensorOrFreeze(ctx);
               TORCHTRT_CHECK(
                   args[1].IValue()->isNone(),
                   "aten::linalg_norm converter does not yet support non-None 'ord' arguments.");
               auto keep_dims = args[3].unwrapToBool();
               auto self_nb_dims = self->getDimensions().nbDims;

               if (!args.back().IValue()->isNone()) {
                 // If specified, the input tensor is cast to dtype before performing the operation, and the returned
                 // tensor’s type will be dtype
                 auto dtype = args.back().unwrapToScalar().to<int64_t>();
                 auto trt_dtype = util::ScalarTypeToTRTDataType(static_cast<at::ScalarType>(dtype));
                 self = castITensor(ctx, self, trt_dtype);
               }

               int32_t axes_mask = 0;
               if (args[2].IValue()->isNone()) {
                 // If dim= None and ord= None, self will be flattened to 1D and the 2-norm of the resulting vector will
                 // be computed.
                 axes_mask = 1;
                 keep_dims = true; // the single output dim is always preserved
                 auto flatten_layer = ctx->net->addShuffle(*self);
                 TORCHTRT_CHECK(flatten_layer, "Unable to create shuffle layer from node: " << *n);
                 flatten_layer->setReshapeDimensions(util::toDims(std::vector<int64_t>({-1})));
                 flatten_layer->setName((util::node_info(n) + "_flatten").c_str());
                 self = flatten_layer->getOutput(0);
               } else {
                 axes_mask = axes_mask_from_axes_values(n, self_nb_dims, args[2].unwrapToIntList().vec());
               }
               auto norm = frobenius_norm(ctx, n, self, axes_mask, keep_dims);
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], norm);
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
