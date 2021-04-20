#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "plugins/interpolate_plugin.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool GlobalPoolingConverter(ConversionCtx* ctx, const torch::jit::Node* n, args& args, nvinfer1::PoolingType pool_type)
{
  auto in = args[0].ITensorOrFreeze(ctx);
  nvinfer1::Dims dims = in->getDimensions();
  // Generate a bitmask of all 1s except the last 2 bits (N and C axes)
  uint32_t reduceAxes = ((1 << dims.nbDims) - 1) & ~0b11;
  auto* new_layer = ctx->net->addReduce(*in, pool_type == nvinfer1::PoolingType::kMAX ? nvinfer1::ReduceOperation::kMAX : nvinfer1::ReduceOperation::kAVG , reduceAxes, /*keepDimensions=*/true);

  new_layer->setName(util::node_info(n).c_str());

  auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

  LOG_DEBUG("GlobalPoolingConverter: Output tensor shape: " << out_tensor->getDimensions());
  return true;
}

bool AdaptivePoolingConverter(ConversionCtx* ctx, const torch::jit::Node* n, args& args,  nvinfer1::PoolingType pool_type) {
  auto in = args[0].ITensorOrFreeze(ctx);
  auto out_size = util::toDims(args[1].unwrapToIntList());
  bool shuffle_back = false;

  // Corner case: when out dimension is all ones, replace with simpler operation
  if (out_size.d[0] == 1 && (out_size.nbDims < 2 || out_size.d[1] == 1 ) && (out_size.nbDims < 3 || out_size.d[2] == 1 ))  {
    return GlobalPoolingConverter(ctx, n, args, pool_type);
  }

  auto shuffle = addPaddingLayer(ctx, n, in, 4, false, false);
  if (shuffle) {
    in = shuffle->getOutput(0);
  }
  
  if (out_size.nbDims == 1) {
    out_size = util::unsqueezeDims(out_size, 0, 1);
    shuffle_back = true;
  }
  auto in_shape = util::toVec(in->getDimensions());
  nvinfer1::ILayer* new_layer = nullptr;

  if (ctx->input_is_dynamic) {
#if NV_TENSORRT_MAJOR < 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR < 1)
    LOG_WARNING(
        "Adaptive pooling layer will be run through ATen, via not TensorRT, performace will be lower than expected. Consider switching either to static input shape or moving to non adaptive pooling if this is an issue");
#else
    LOG_WARNING(
        "Adaptive pooling layer will be run through ATen (on CPU), via not TensorRT, performace will suffer. Consider switching either to static input shape or moving to non adaptive pooling");
#endif
    
    TRTORCH_CHECK(pool_type == nvinfer1::PoolingType::kAVERAGE,
		  "Unable to create MAX pooling (interpolation) plugin from node" << *n);

    auto out_shape = in_shape;
    std::copy_n(out_size.d, out_size.nbDims, out_shape.begin() + (in_shape.size() - out_size.nbDims));

    auto creator = new plugins::InterpolatePluginCreator();
    auto plugin = creator->createPlugin("adaptive_pool2d", in_shape, out_shape,
					util::toVec(out_size), {}, std::string("adaptive_pool2d"), false, false);

    new_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
    TRTORCH_CHECK(new_layer, "Unable to create pooling (interpolation) plugin from node" << *n);

  } else {
    std::vector<int64_t> stride(out_size.nbDims);
    for (size_t i = 0; i < out_size.nbDims; i++) {
      stride[(stride.size() - 1) - i] = in_shape[(in_shape.size() - 1) - i] / out_size.d[(out_size.nbDims - 1) - i];
    }
    LOG_DEBUG("Stride: " << util::toDims(stride));

    std::vector<int64_t> window(out_size.nbDims);
    for (size_t i = 0; i < out_size.nbDims; i++) {
      window[window.size() - 1 - i] =
          in_shape[in_shape.size() - 1 - i] - (out_size.d[out_size.nbDims - 1 - i] - 1) * stride[stride.size() - 1 - i];
    }

    LOG_DEBUG("Window: " << util::toDims(window));

    auto pooling_layer = ctx->net->addPoolingNd(*in, pool_type, util::toDims(window));
    TRTORCH_CHECK(pooling_layer, "Unable to create average pooling layer from node: " << *n);
    pooling_layer->setStrideNd(util::toDims(stride));
    new_layer = pooling_layer;
  }

  new_layer->setName(util::node_info(n).c_str());

  if (shuffle_back ) {
    new_layer = addUnpaddingLayer(ctx, n, new_layer->getOutput(0), 3, false, false);
  }

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));
  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());

  return true;
}
  
bool PoolingConverter(ConversionCtx* ctx, const torch::jit::Node* n, args& args, nvinfer1::PoolingType pool_type) {
  auto in = args[0].ITensorOrFreeze(ctx);
  
  // Max Pool needs at least 4D input
  auto shuffle = addPaddingLayer(ctx, n, in, 4, false, true);

  if (shuffle) {
    in = shuffle->getOutput(0);
  }

  auto kernel_size = util::toDims(args[1].unwrapToIntList());
  auto padding = util::toDims(args[3].unwrapToIntList());
  auto stride = util::toDims(args[2].unwrapToIntList());
  if (stride.nbDims == 0) {
    LOG_DEBUG("Stride not providied, using kernel_size as stride");
    stride = util::toDims(args[1].unwrapToIntList());
  }

  bool shuffle_back = false;
  if (kernel_size.nbDims == 1) {
    kernel_size = util::unsqueezeDims(kernel_size, 0, 1);
    if (shuffle)
      shuffle_back = true;
    LOG_DEBUG("kernel_size.nbDims < 2, padding:" << kernel_size);
    LOG_DEBUG("kernel_size: " << kernel_size);
  }
  if (padding.nbDims == 1)
    padding = util::unsqueezeDims(padding, 0, 0); 
  if (stride.nbDims == 1)
    stride = util::unsqueezeDims(stride, 0, 1);

  LOG_DEBUG("kernel_size: " << kernel_size);
  LOG_DEBUG("padding: " << padding);
  LOG_DEBUG("stride: " << stride);

  bool ceil_mode;
  nvinfer1::IPoolingLayer* new_layer;

  if (pool_type == nvinfer1::PoolingType::kMAX) {
    auto dilation = util::toDims(args[4].unwrapToIntList());

    TRTORCH_ASSERT(
        dilation == util::toDims(std::vector<int64_t>(dilation.nbDims, 1)),
        "Pooling dilation is not supported in TensorRT");

    LOG_DEBUG("dilation: " << dilation);
    LOG_WARNING("Dilation not used in Max pooling converter");
    ceil_mode = args[5].unwrapToBool();

    new_layer = ctx->net->addPoolingNd(*in, pool_type, kernel_size);
    TRTORCH_CHECK(new_layer, "Unable to create Max Pooling layer from node: " << *n);
  } else if (pool_type == nvinfer1::PoolingType::kAVERAGE) {
    ceil_mode = args[4].unwrapToBool();
    bool count_inlcude_pad = args[5].unwrapToBool();

    new_layer = ctx->net->addPoolingNd(*in, pool_type, kernel_size);
    TRTORCH_CHECK(new_layer, "Unable to create Avg Pooling layer from node: " << *n);
    new_layer->setAverageCountExcludesPadding(!count_inlcude_pad);
    // if (!(args[6].IValue()->isNone())) {
    //  LOG_WARNING("Divisor override is now handled by Avg Pooling Converter");
    // }
  } else {
    TRTORCH_ASSERT(0, "Unsupported pool mode!");
  }

  new_layer->setName(util::node_info(n).c_str());
  new_layer->setPaddingNd(padding);
  if (stride.nbDims != 2 && ctx->settings.device.device_type == nvinfer1::DeviceType::kDLA) {
    if (!ctx->settings.device.allow_gpu_fallback) {
      TRTORCH_THROW_ERROR("DLA Pooling stride is limited to 2D, allow GPU fallback");
    } else {
      LOG_WARNING("DLA Pooling stride is limited to 2D, will run on GPU");
    }
  }
  new_layer->setStrideNd(stride);

  auto padding_mode =
      ceil_mode ? nvinfer1::PaddingMode::kEXPLICIT_ROUND_UP : nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN;
  new_layer->setPaddingMode(padding_mode);

  new_layer->setName(util::node_info(n).c_str());

  nvinfer1::ILayer* out_layer = new_layer;
  
  if (shuffle_back) {
    out_layer = addUnpaddingLayer(ctx, n, new_layer->getOutput(0), 3, false, true);
  }

  auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_layer->getOutput(0));

  LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
  return true;
} // namespace

auto pooling_registrations TRTORCH_UNUSED =
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
        .pattern({"aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
	       return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE);
	    }})
        .pattern({"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
	       return AdaptivePoolingConverter(ctx, n, args, nvinfer1::PoolingType::kAVERAGE);
	    }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
