#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool GlobalPoolingConverter(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    args& args,
    nvinfer1::PoolingType pool_type) {
  auto in = args[0].ITensorOrFreeze(ctx);
  nvinfer1::Dims dims = in->getDimensions();
  // Generate a bitmask of all 1s except the last 2 bits (N and C axes) when dims.nbDims > 2
  uint32_t reduceAxes = ((1 << dims.nbDims) - 1) & ~0b11;
  // Generate a bitmask of all 1s except the last 1 bits (N axes) when dims.nbDims == 2. `aten::adaptive_avg_pool1d`'s
  // input can be (N, C, L) or (C, L).
  if (dims.nbDims == 2) {
    reduceAxes = ((1 << dims.nbDims) - 1) & ~0b1;
  }
  auto* new_layer = ctx->net->addReduce(
      *in,
      pool_type == nvinfer1::PoolingType::kMAX ? nvinfer1::ReduceOperation::kMAX : nvinfer1::ReduceOperation::kAVG,
      reduceAxes,
      /*keepDimensions=*/true);

  new_layer->setName(util::node_info(n).c_str());

  auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

  LOG_DEBUG("GlobalPoolingConverter: Output tensor shape: " << out_tensor->getDimensions());
  return true;
}

bool AdaptivePoolingConverter(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    args& args,
    nvinfer1::PoolingType pool_type,
    const std::string& mode) {
  auto in = args[0].ITensorOrFreeze(ctx);
  auto out_size = util::toDims(args[1].unwrapToIntList());

  // Corner case: when out dimension is all ones, replace with simpler operation
  if (out_size.d[0] == 1 && (out_size.nbDims < 2 || out_size.d[1] == 1) &&
      (out_size.nbDims < 3 || out_size.d[2] == 1)) {
    return GlobalPoolingConverter(ctx, n, args, pool_type);
  }

  auto orig_dims = in->getDimensions();
  TORCHTRT_CHECK(orig_dims.nbDims > 1, "Unable to create pooling layer from node: " << *n);

  auto in_shape = util::toVec(in->getDimensions());
  nvinfer1::ILayer* new_layer = nullptr;

  /*======CONFIGURE PLUGIN PARAMETERS======*/
  nvinfer1::PluginFieldCollection fc;
  std::vector<nvinfer1::PluginField> f;

  auto out_shape = in_shape;
  auto out_size_vec = util::toVec(out_size);

  std::copy(out_size_vec.begin(), out_size_vec.end(), out_shape.begin() + (in_shape.size() - out_size_vec.size()));

  std::vector<int32_t> in_shape_casted(in_shape.begin(), in_shape.end());
  f.emplace_back(
      nvinfer1::PluginField("in_shape", in_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, in_shape.size()));

  std::vector<int32_t> out_shape_casted(out_shape.begin(), out_shape.end());
  f.emplace_back(
      nvinfer1::PluginField("out_shape", out_shape_casted.data(), nvinfer1::PluginFieldType::kINT32, out_shape.size()));

  std::vector<int32_t> out_size_casted(out_size_vec.begin(), out_size_vec.end());
  f.emplace_back(nvinfer1::PluginField(
      "out_size", out_size_casted.data(), nvinfer1::PluginFieldType::kINT32, out_size_vec.size()));

  f.emplace_back(nvinfer1::PluginField("scales", nullptr, nvinfer1::PluginFieldType::kFLOAT64, 0));

  int32_t align_corners_casted = 0;
  f.emplace_back(nvinfer1::PluginField("align_corners", &align_corners_casted, nvinfer1::PluginFieldType::kINT32, 1));

  int32_t use_scales_casted = 0;
  f.emplace_back(nvinfer1::PluginField("use_scales", &use_scales_casted, nvinfer1::PluginFieldType::kINT32, 1));

  f.emplace_back(nvinfer1::PluginField("mode", &mode, nvinfer1::PluginFieldType::kCHAR, 1));

  fc.nbFields = f.size();
  fc.fields = f.data();
  /*====== PLUGIN PARAMETERS CONFIGURATION COMPLETED ======*/

  LOG_WARNING(
      "Adaptive pooling layer will be using Aten library kernels in pytorch for execution. TensorRT does not support adaptive pooling natively. Consider switching to non-adaptive pooling if this is an issue");

  auto creator = getPluginRegistry()->getPluginCreator("Interpolate", "1", "torch_tensorrt");
  auto interpolate_plugin = creator->createPlugin(mode.c_str(), &fc);

  new_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *interpolate_plugin);
  TORCHTRT_CHECK(new_layer, "Unable to create pooling (interpolation) plugin from node" << *n);

  new_layer->setName(util::node_info(n).c_str());
  auto layer_output = new_layer->getOutput(0);

  ctx->AssociateValueAndTensor(n->outputs()[0], layer_output);
  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());

  return true;
}

bool PoolingConverter(ConversionCtx* ctx, const torch::jit::Node* n, args& args, nvinfer1::PoolingType pool_type) {
  auto in = args[0].ITensorOrFreeze(ctx);

  // Max Pool needs at least 4D input
  auto orig_dims = in->getDimensions();
  TORCHTRT_CHECK(orig_dims.nbDims > 2, "Unable to create pooling layer from node: " << *n);
  bool expandDims = (orig_dims.nbDims < 4);

  if (expandDims) {
    in = addPadding(ctx, n, in, 4, false, true);
  }

  auto kernel_size = util::toDims(args[1].unwrapToIntList());
  auto padding = util::toDims(args[3].unwrapToIntList());
  auto stride = util::toDims(args[2].unwrapToIntList());
  if (stride.nbDims == 0) {
    LOG_DEBUG("Stride not provided, using kernel_size as stride");
    stride = util::toDims(args[1].unwrapToIntList());
  }

  if (kernel_size.nbDims == 1) {
    kernel_size = util::unsqueezeDims(kernel_size, 0, 1);
    LOG_DEBUG("kernel_size.nbDims < 2, padding:" << kernel_size);
    LOG_DEBUG("kernel_size: " << kernel_size);
  }
  if (padding.nbDims == 1) {
    padding = util::unsqueezeDims(padding, 0, 0);
  }
  if (stride.nbDims == 1) {
    stride = util::unsqueezeDims(stride, 0, 1);
  }

  LOG_DEBUG("kernel_size: " << kernel_size);
  LOG_DEBUG("padding: " << padding);
  LOG_DEBUG("stride: " << stride);

  bool ceil_mode;
  nvinfer1::IPoolingLayer* new_layer;

  if (pool_type == nvinfer1::PoolingType::kMAX) {
    auto dilation = util::toDims(args[4].unwrapToIntList());

    TORCHTRT_CHECK(
        dilation == util::toDims(std::vector<int64_t>(dilation.nbDims, 1)),
        "Pooling dilation is not supported in TensorRT");

    LOG_DEBUG("dilation: " << dilation);
    LOG_WARNING("Dilation not used in Max pooling converter");
    ceil_mode = args[5].unwrapToBool();

    new_layer = ctx->net->addPoolingNd(*in, pool_type, kernel_size);
    TORCHTRT_CHECK(new_layer, "Unable to create Max Pooling layer from node: " << *n);
  } else if (pool_type == nvinfer1::PoolingType::kAVERAGE) {
    ceil_mode = args[4].unwrapToBool();
    bool count_inlcude_pad = args[5].unwrapToBool();

    new_layer = ctx->net->addPoolingNd(*in, pool_type, kernel_size);
    TORCHTRT_CHECK(new_layer, "Unable to create Avg Pooling layer from node: " << *n);
    new_layer->setAverageCountExcludesPadding(!count_inlcude_pad);
  } else {
    TORCHTRT_THROW_ERROR("Unsupported pool mode!");
  }

  auto padding_mode =
      ceil_mode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;

  new_layer->setName(util::node_info(n).c_str());
  new_layer->setPaddingMode(padding_mode);
  new_layer->setPaddingNd(padding);
  new_layer->setStrideNd(stride);

  if (stride.nbDims != 2 && ctx->settings.device.device_type == nvinfer1::DeviceType::kDLA) {
    if (!ctx->settings.device.allow_gpu_fallback) {
      TORCHTRT_THROW_ERROR("DLA Pooling stride is limited to 2D, allow GPU fallback");
    } else {
      LOG_WARNING("DLA Pooling stride is limited to 2D, will run on GPU");
    }
  }

  auto out_tensor = addUnpadding(ctx, n, new_layer->getOutput(0), orig_dims.nbDims, false, true);
  ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);

  LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
  return true;
} // namespace

auto pooling_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[], int[1] dilation=[], bool ceil_mode=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return PoolingConverter(ctx, n, args, nvinfer1::PoolingType::kMAX);
             }})
        .pattern(
            {"aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return PoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE);
             }})
        .pattern(
            {"aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], int[2] dilation=[1, 1], bool ceil_mode=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return PoolingConverter(ctx, n, args, nvinfer1::PoolingType::kMAX);
             }})
        .pattern(
            {"aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=[0, 0], bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return PoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE);
             }})
        .pattern(
            {"aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=[], int[3] dilation=[], bool ceil_mode=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return PoolingConverter(ctx, n, args, nvinfer1::PoolingType::kMAX);
             }})
        .pattern(
            {"aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=[], bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return PoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE);
             }})
        .pattern(
            {"aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE, "adaptive_avg_pool1d");
             }})
        .pattern(
            {"aten::adaptive_max_pool1d(Tensor self, int[2] output_size) -> (Tensor, Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kMAX, "adaptive_max_pool1d");
             }})
        .pattern(
            {"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE, "adaptive_avg_pool2d");
             }})
        .pattern(
            {"aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kMAX, "adaptive_max_pool2d");
             }})
        .pattern(
            {"aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE, "adaptive_avg_pool3d");
             }})
        .pattern(
            {"aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kMAX, "adaptive_max_pool3d");
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
