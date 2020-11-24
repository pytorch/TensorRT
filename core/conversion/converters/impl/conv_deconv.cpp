#include "torch/torch.h"

#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {
auto conv_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::_convolution(Tensor input, Tensor weight,
                                 Tensor? bias, int[] stride, int[] padding,
                                 int[] dilation, bool transposed,
                                 int[] output_padding, int groups, bool benchmark,
                                 bool deterministic, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto in = args[0].ITensor(); // assumes non-static input Tensor
      auto w = Weights(ctx, args[1].unwrapToTensor());
      auto stride = util::toDims(args[3].unwrapToIntList());
      LOG_DEBUG("stride: " << stride);
      auto padding = util::toDims(args[4].unwrapToIntList());
      LOG_DEBUG("padding: " << padding);
      auto dilation = util::toDims(args[5].unwrapToIntList());
      LOG_DEBUG("dilation: " << dilation);
      bool transposed = args[6].unwrapToBool();
      auto out_padding = util::toDims(args[7].unwrapToIntList());
      LOG_DEBUG("out_padding: " << out_padding);
      int64_t groups = args[8].unwrapToInt();
      LOG_DEBUG("groups: " << groups);

      nvinfer1::ILayer* new_layer;
      if (transposed) {
        Weights bias;
        if (args[2].IValue()->isTensor()) {
          bias = Weights(ctx, args[2].unwrapToTensor());
        } else {
          bias = Weights(ctx, torch::zeros(args[1].unwrapToTensor().sizes()[1] * groups));
        }

        auto deconv = ctx->net->addDeconvolutionNd(*in, w.num_input_maps * groups, w.kernel_shape, w.data, bias.data);
        TRTORCH_CHECK(deconv, "Unable to create deconvolution layer from node: " << *n);

        deconv->setStrideNd(stride);
        deconv->setPaddingNd(padding);
#if NV_TENSORRT_MAJOR > 7 || (NV_TENSORRT_MAJOR == 7 && NV_TENSORRT_MINOR == 1)
        deconv->setDilationNd(dilation);
        deconv->setNbGroups(groups);
#else
        TRTORCH_CHECK(groups == 1, "for deconv with groups > 1, require TensorRT version >= 7.1");
        for (auto it = dilation.begin(); it != dilation.end(); ++it) {
          TRTORCH_CHECK(*it == 1, "for deconv with dilation > 1, require TensorRT version >= 7.1");
        }
#endif
        new_layer = deconv;
      } else {
        Weights bias;
        if (args[2].IValue()->isTensor()) {
          bias = Weights(ctx, args[2].unwrapToTensor());
        } else {
          bias = Weights(ctx, torch::zeros(args[1].unwrapToTensor().sizes()[0]));
        }

        auto conv = ctx->net->addConvolutionNd(*in, w.num_output_maps, w.kernel_shape, w.data, bias.data);
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

      auto out = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));

      LOG_DEBUG("Output tensor shape: " << out->getDimensions());

      return true;
    }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
