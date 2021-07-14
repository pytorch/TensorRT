#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool add_conv_deconv(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {
  // Input to conv/deconv
  auto in = args[0].ITensor();

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
    bias = Weights(); // nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};
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
    int64_t nbSpatialDims = in->getDimensions().nbDims - 2;
    TRTORCH_CHECK(
        nbSpatialDims = kernel_dims.nbDims - 2,
        "Number of input spatial dimensions should match the kernel spatial dimensions");
    filter_dim.nbDims = nbSpatialDims;
    filter_dim.d[0] = kernel_dims.d[2];
    filter_dim.d[1] = kernel_dims.d[3];

    // Initialize a dummy constant kernel to pass it to INetwork->addConvolutionNd/addDeconvolutionNd API.
    auto kernel_weights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, nullptr, 0};

    nvinfer1::ILayer* layer = nullptr;
    if (transposed) {
      nvinfer1::IDeconvolutionLayer* deconvLayer =
          ctx->net->addDeconvolutionNd(*in, kernel_dims.d[0], filter_dim, kernel_weights, bias.data);
      deconvLayer->setStrideNd(stride);
      deconvLayer->setDilationNd(dilation);
      deconvLayer->setNbGroups(groups);
      deconvLayer->setPaddingNd(padding);
      // Set deconv kernel weights
      deconvLayer->setInput(1, *kernel);
      TRTORCH_CHECK(deconvLayer, "Unable to create deconv layer with non-const weights from node: " << *n);
      layer = deconvLayer;
    } else {
      nvinfer1::IConvolutionLayer* convLayer =
          ctx->net->addConvolutionNd(*in, kernel_dims.d[0], filter_dim, kernel_weights, bias.data);
      convLayer->setStrideNd(stride);
      convLayer->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN);
      convLayer->setPaddingNd(padding);
      convLayer->setPostPadding(out_padding);
      convLayer->setDilationNd(dilation);
      convLayer->setNbGroups(groups);

      // Set conv kernel weights
      convLayer->setInput(1, *kernel);
      layer = convLayer;
    }

    ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
    LOG_DEBUG("Output tensor shape: " << layer->getOutput(0)->getDimensions());
    return true;
  }

  auto w = Weights(ctx, args[1].unwrapToTensor());
  auto dims = in->getDimensions();
  auto orig_dims = dims;
  LOG_DEBUG("Input dims: " << orig_dims);
  LOG_DEBUG("Weights: " << w);
  LOG_DEBUG("stride: " << stride);
  LOG_DEBUG("padding: " << padding);
  LOG_DEBUG("dilation: " << dilation);
  LOG_DEBUG("out_padding: " << out_padding);
  LOG_DEBUG("groups: " << groups);

  // Expand spatial dims from 1D to 2D if needed
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
    // shape of deconvolution's weight: [in, out/groups, ...]
    auto deconv = ctx->net->addDeconvolutionNd(*in, w.shape.d[1] * groups, w.kernel_shape, w.data, bias.data);
    TRTORCH_CHECK(deconv, "Unable to create deconvolution layer from node: " << *n);

    deconv->setStrideNd(stride);
    deconv->setPaddingNd(padding);
#if NV_TENSORRT_MAJOR > 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR >= 1)
    deconv->setDilationNd(dilation);
    deconv->setNbGroups(groups);
#else
    TRTORCH_CHECK(groups == 1, "for deconv with groups > 1, require TensorRT version >= 7.1");
    for (int idx = 0; idx < dilation.nbDims; idx++) {
      TRTORCH_CHECK(dilation.d[idx] == 1, "for deconv with dilation > 1, require TensorRT version >= 7.1");
    }
#endif
    new_layer = deconv;
  } else {
    // Weights bias;
    // if (args[2].IValue()->isTensor()) {
    //   bias = Weights(ctx, args[2].unwrapToTensor());
    // } else {
    //   bias = Weights(ctx, torch::zeros(w.shape.d[0]));
    // }

    // shape of convolution's weight: [out, in/groups, ...]
    auto conv = ctx->net->addConvolutionNd(*in, w.shape.d[0], w.kernel_shape, w.data, bias.data);
    TRTORCH_CHECK(conv, "Unable to create convolution layer from node: " << *n);

    conv->setStrideNd(stride);
    conv->setPaddingMode(nvinfer1::PaddingMode::kCAFFE_ROUND_DOWN);
    conv->setPaddingNd(padding);
    conv->setPostPadding(out_padding);
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

auto conv_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
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
} // namespace trtorch
