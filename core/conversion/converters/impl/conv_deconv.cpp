#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

void add_output_padding(nvinfer1::Dims& padding, nvinfer1::Dims& out_padding, bool& has_output_padding) {
  int nbSpatialDims = out_padding.nbDims;
  // When there is out_padding, if padding is larger than out_padding, just adjust padding Or reduce out_padding as
  // minimum as possible.
  for (int i = 0; i < nbSpatialDims; ++i) {
    if (padding.d[i] - out_padding.d[i] >= 0) {
      padding.d[i] -= out_padding.d[i];
      out_padding.d[i] = 0;
    } else {
      // Reduce out_padding as possible.
      out_padding.d[i] -= padding.d[i];
      padding.d[i] = 0;
      has_output_padding = true;
    }
  }
}

nvinfer1::ILayer* add_bias_layer(
    ConversionCtx* ctx,
    nvinfer1::ITensor* input_tensor,
    nvinfer1::Dims& input_dims,
    nvinfer1::Dims& output_padding,
    Weights& bias) {
  nvinfer1::ITensor* input_shape = getShapeOutput(ctx, input_tensor, std::string("bias_shape_cast").c_str());
  // Add padding layer
  nvinfer1::ITensor* start;
  nvinfer1::ITensor* totalPadding;
  auto in_nbDims = input_dims.nbDims;
  std::vector<int32_t> startVec(in_nbDims, 0);
  std::vector<int32_t> totalPaddingVec(in_nbDims, 0);
  int32_t diff = in_nbDims - output_padding.nbDims;
  for (int32_t i = diff; i < in_nbDims; i++) {
    int32_t idx = i - diff;
    startVec[i] = 0; // Don't need begin padding, only post padding
    totalPaddingVec[i] = output_padding.d[idx];
  }
  start = tensor_to_const(ctx, torch::tensor(startVec, torch::kInt32));
  totalPadding = tensor_to_const(ctx, torch::tensor(totalPaddingVec, torch::kInt32));

  const auto size =
      ctx->net->addElementWise(*input_shape, *totalPadding, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);

  nvinfer1::Dims stride;
  stride.nbDims = in_nbDims;
  for (int64_t i = 0; i < in_nbDims; i++) {
    stride.d[i] = 1;
  }
  const auto& dummy = stride;
  auto* sliceLayer = ctx->net->addSlice(*input_tensor, dummy, dummy, stride);
  sliceLayer->setInput(1, *start);
  sliceLayer->setInput(2, *size);
  sliceLayer->setMode(nvinfer1::SampleMode::kFILL);
  nvinfer1::ITensor* slice_output = sliceLayer->getOutput(0);

  nvinfer1::Dims constantDims;
  constantDims.nbDims = in_nbDims;
  for (int64_t i = 0; i < in_nbDims; i++) {
    constantDims.d[i] = 1;
  }
  constantDims.d[diff - 1] =
      bias.shape.d[0]; // Set C dimension to bias dim and other dimensions to 1 to enable broadcast
  auto const_layer = ctx->net->addConstant(constantDims, bias.data);
  auto bias_layer =
      ctx->net->addElementWise(*slice_output, *const_layer->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

  return bias_layer;
}

bool add_conv_deconv(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {
  // Input to conv/deconv
  auto in = args[0].ITensor();
  if (in->getType() == nvinfer1::DataType::kINT32) {
    LOG_WARNING(
        "Found type  " << in->getType() << "in aten::convolution, casting to" << nvinfer1::DataType::kFLOAT
                       << " for compatibility.");
    in = castITensor(ctx, in, nvinfer1::DataType::kFLOAT);
  }
  // Conv /deconv parameters
  auto stride = util::toDims(args[3].unwrapToIntList());
  auto padding = util::toDims(args[4].unwrapToIntList());
  auto dilation = util::toDims(args[5].unwrapToIntList());
  bool transposed = args[6].unwrapToBool();
  auto out_padding = util::toDims(args[7].unwrapToIntList());
  int64_t groups = args[8].unwrapToInt();

  // Reshape the parameters to 2D if needed
  if (stride.nbDims == 1) {
    stride = util::unsqueezeDims(stride, 1, 1);
    LOG_DEBUG("Reshaped stride: " << stride);
  }
  if (dilation.nbDims == 1) {
    dilation = util::unsqueezeDims(dilation, 1, 1);
    LOG_DEBUG("Reshaped dilation: " << dilation);
  }
  if (padding.nbDims == 1) {
    padding = util::unsqueezeDims(padding, 1, 0);
    LOG_DEBUG("Reshaped padding: " << padding);
  }
  if (out_padding.nbDims == 1) {
    out_padding = util::unsqueezeDims(out_padding, 1, 0);
    LOG_DEBUG("Reshaped out_padding: " << out_padding);
  }

  // Get bias tensor or initialize it to zeros.
  Weights bias;
  if (args[2].IValue()->isTensor()) {
    bias = Weights(ctx, args[2].unwrapToTensor());
  } else {
    bias = Weights();
  }

  // Handle case when weights of conv/deconv is an ITensor. This case happens for QAT networks where
  // conv_weights -> Quantize -> Dequantize -> new_conv_weights -> conv <- input
  // new_conv_weights will be an ITensor because it is an output of Dequantize layer defined in impl/quantization.cpp
  if (args[1].isITensor()) {
    // Get the kernel tensor
    auto kernel = args[1].ITensor();
    auto kernel_dims = kernel->getDimensions();

    // Make a new Dims with only the spatial dimensions.
    nvinfer1::Dims filter_dim;
    nvinfer1::Dims original_dim = in->getDimensions();
    int64_t nbSpatialDims = in->getDimensions().nbDims - 2;
    TORCHTRT_CHECK(
        nbSpatialDims = kernel_dims.nbDims - 2,
        "Number of input spatial dimensions should match the kernel spatial dimensions");
    filter_dim.nbDims = nbSpatialDims;
    filter_dim.d[0] = kernel_dims.d[2];
    filter_dim.d[1] = kernel_dims.d[3];
    // For Conv2d layer, weights are in the shape of (out_channels, in_channels/groups,...)
    int32_t num_output_maps = kernel_dims.d[0];
    if (transposed) {
      // For ConvTranspose layer, weights are in the shape of (in_channels, out_channel/groups,...)
      num_output_maps = kernel_dims.d[1];
    }
    bool expand_dims = nbSpatialDims == 1;
    if (expand_dims) {
      // In case of Conv1D -> map it to 2D version
      // TensorRT expects nbSpatialDims = 2 or 3
      filter_dim = util::unsqueezeDims(filter_dim, filter_dim.nbDims, 1, false);
      // Reshape input dimensions
      in = addPadding(ctx, n, in, 4, true, true, std::string(util::node_info(n) + "_input_shuffle"));
      LOG_DEBUG("Reshaping input dimensions to: " << in->getDimensions());
      kernel = addPadding(ctx, n, kernel, 4, true, true, std::string(util::node_info(n) + "_kernel_shuffle"));
      LOG_DEBUG("Reshaping kernel dimensions to: " << kernel->getDimensions());
    }

    // Initialize a dummy constant kernel to pass it to INetwork->addConvolutionNd/addDeconvolutionNd API.
    auto kernel_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ILayer* layer = nullptr;
    nvinfer1::ITensor* out = nullptr;
    if (transposed) {
      // Fix padding based on output_padding provided
      nvinfer1::Dims begPadding = padding;
      bool hasOutputPadding = false;
      add_output_padding(padding, out_padding, hasOutputPadding);

      nvinfer1::IDeconvolutionLayer* deconvLayer = ctx->net->addDeconvolutionNd(
          *in, num_output_maps, filter_dim, kernel_weights, hasOutputPadding ? nvinfer1::Weights{} : bias.data);
      deconvLayer->setStrideNd(stride);
      deconvLayer->setDilationNd(dilation);
      deconvLayer->setNbGroups(groups);
      deconvLayer->setPrePadding(begPadding);
      deconvLayer->setPostPadding(padding);

      // Set deconv kernel weights
      deconvLayer->setInput(1, *kernel);
      TORCHTRT_CHECK(deconvLayer, "Unable to create deconv layer with non-const weights from node: " << *n);
      layer = deconvLayer;
      out = deconvLayer->getOutput(0);
      if (hasOutputPadding) {
        LOG_DEBUG("Padding output deconvolution tensor with:" << out_padding);
        nvinfer1::ITensor* tensorPtr = deconvLayer->getOutput(0);
        auto dims = in->getDimensions();
        layer = add_bias_layer(ctx, tensorPtr, dims, out_padding, bias);
        out = layer->getOutput(0);
      }
      if (expand_dims) {
        // Un-expand the expanded dimension
        out = addUnpadding(ctx, n, out, original_dim.nbDims);
      }
    } else {
      nvinfer1::IConvolutionLayer* convLayer =
          ctx->net->addConvolutionNd(*in, num_output_maps, filter_dim, kernel_weights, bias.data);
      convLayer->setStrideNd(stride);
      convLayer->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
      convLayer->setPaddingNd(padding);
      convLayer->setPostPadding(out_padding);
      convLayer->setDilationNd(dilation);
      convLayer->setNbGroups(groups);

      // Set conv kernel weights
      convLayer->setInput(1, *kernel);
      layer = convLayer;
      out = layer->getOutput(0);
    }

    ctx->AssociateValueAndTensor(n->outputs()[0], out);
    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
    return true;
  }

  auto w = Weights(ctx, args[1].unwrapToTensor());
  // TODO: Remove this when conv3d with kernel size=1 bug is fixed.
  // Github issue: https://github.com/pytorch/TensorRT/issues/1445
  bool is_kernel_size_one = true;
  bool is_3d_kernel = w.kernel_shape.nbDims == 3;
  for (int64_t i = 0; i < w.kernel_shape.nbDims; i++) {
    if (w.kernel_shape.d[i] != 1.0f) {
      is_kernel_size_one = false;
    }
  }
  if (is_kernel_size_one && is_3d_kernel) {
    LOG_WARNING(
        "Conv3d layer with kernel size = 1 configuration incurs a failure with TensorRT tactic optimizer in some cases. \
    Github issue: https://github.com/pytorch/TensorRT/issues/1445. Other conv variants do not have this issue.");
  }
  auto dims = in->getDimensions();
  auto orig_dims = dims;
  LOG_DEBUG("Input dims: " << orig_dims);
  LOG_DEBUG("Weights: " << w);
  LOG_DEBUG("stride: " << stride);
  LOG_DEBUG("padding: " << padding);
  LOG_DEBUG("dilation: " << dilation);
  LOG_DEBUG("out_padding: " << out_padding);
  LOG_DEBUG("groups: " << groups);

  TORCHTRT_CHECK(orig_dims.nbDims > 2, "Unable to create convolution layer from node: " << *n);

  bool expandDims = (orig_dims.nbDims < 4);
  if (expandDims) {
    in = addPadding(ctx, n, in, 4);
    dims = in->getDimensions();
    LOG_DEBUG("Reshaped Input dims: " << dims);
  }
  if (w.shape.nbDims < 4) {
    for (int i = w.shape.nbDims; i < 4; ++i) {
      w.shape.d[i] = 1;
    }
    w.shape.nbDims = 4;
    w.kernel_shape.nbDims = 2;
    w.kernel_shape.d[1] = 1;
    LOG_DEBUG("Reshaped Weights: " << w);
  }

  nvinfer1::ILayer* new_layer;
  if (transposed) {
    // Refer to
    // https://github.com/onnx/onnx-tensorrt/blob/c3cfcbc8248c6bd007e6630af2085df5e4834b42/builtin_op_importers.cpp#L734
    nvinfer1::Dims begPadding = padding;
    bool hasOutputPadding = false;
    add_output_padding(padding, out_padding, hasOutputPadding);

    // shape of deconvolution's weight: [in, out/groups, ...]
    // If there is still output padding, remove the bias. Bias will be added below.
    auto deconv = ctx->net->addDeconvolutionNd(
        *in, w.shape.d[1] * groups, w.kernel_shape, w.data, hasOutputPadding ? nvinfer1::Weights{} : bias.data);
    TORCHTRT_CHECK(deconv, "Unable to create deconvolution layer from node: " << *n);

    deconv->setStrideNd(stride);
    deconv->setPrePadding(begPadding);
    deconv->setPostPadding(padding);
#if NV_TENSORRT_MAJOR > 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR >= 1)
    deconv->setDilationNd(dilation);
    deconv->setNbGroups(groups);
#else
    TORCHTRT_CHECK(groups == 1, "for deconv with groups > 1, require TensorRT version >= 7.1");
    for (int idx = 0; idx < dilation.nbDims; idx++) {
      TORCHTRT_CHECK(dilation.d[idx] == 1, "for deconv with dilation > 1, require TensorRT version >= 7.1");
    }
#endif
    if (hasOutputPadding) {
      LOG_DEBUG("Padding output deconvolution tensor with:" << out_padding);
      nvinfer1::ITensor* tensorPtr = deconv->getOutput(0);
      new_layer = add_bias_layer(ctx, tensorPtr, orig_dims, out_padding, bias);
    } else {
      new_layer = deconv;
    }
  } else {
    // shape of convolution's weight: [out, in/groups, ...]
    auto conv = ctx->net->addConvolutionNd(*in, w.shape.d[0], w.kernel_shape, w.data, bias.data);
    TORCHTRT_CHECK(conv, "Unable to create convolution layer from node: " << *n);
    conv->setStrideNd(stride);
    conv->setPaddingMode(nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN);
    conv->setPaddingNd(padding);
    conv->setDilationNd(dilation);
    conv->setNbGroups(groups);
    new_layer = conv;
  }

  new_layer->setName(util::node_info(n).c_str());

  // Un-expand spatial dims back to 1D if needed
  auto out = addUnpadding(ctx, n, new_layer->getOutput(0), orig_dims.nbDims);

  ctx->AssociateValueAndTensor(n->outputs()[0], out);

  LOG_DEBUG("Output tensor shape: " << out->getDimensions());

  return true;
}

auto conv_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({
            R"SIG(aten::_convolution(Tensor input, Tensor weight,
                                 Tensor? bias, int[] stride, int[] padding,
                                 int[] dilation, bool transposed,
                                 int[] output_padding, int groups, bool benchmark,
                                 bool deterministic, bool cudnn_enabled, bool allow_tf32) -> (Tensor))SIG",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              return add_conv_deconv(ctx, n, args);
            }})
        .pattern({
            R"SIG(aten::_convolution.deprecated(Tensor input, Tensor weight,
                                     Tensor? bias, int[] stride, int[] padding,
                                     int[] dilation, bool transposed,
                                     int[] output_padding, int groups, bool benchmark,
                                     bool deterministic, bool cudnn_enabled) -> (Tensor))SIG",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              // This pattern is only matched for traced JIT models which do not
              // have allow_tf32 bool in the function signature. The TRT conversion
              // code is exactly same as the above call.
              return add_conv_deconv(ctx, n, args);
            }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
