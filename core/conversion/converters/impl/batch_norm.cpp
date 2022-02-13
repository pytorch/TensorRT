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

void _batch_norm(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* input,
    const nvinfer1::Dims& orig_shape,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& mean,
    const torch::Tensor& var,
    const float eps) {
  auto scale = gamma / torch::sqrt(var + eps);
  auto bias = beta - mean * scale;
  LOG_DEBUG("_batch_norm Tensor Scale : " << scale.sizes());
  LOG_DEBUG("_batch_norm Tensor bias : " << bias.sizes());

  auto scale_weights = Weights(ctx, scale);
  auto bias_weights = Weights(ctx, bias);

  auto power = Weights(ctx, at::ones_like(scale));
  auto bn =
      ctx->net->addScaleNd(*input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
  bn->setName(util::node_info(n).c_str());

  // Un-pad bn output if needed
  auto out_tensor = addUnpadding(ctx, n, bn->getOutput(0), orig_shape.nbDims);
  ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
  LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
}

auto batch_norm_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({
            R"SIG(aten::batch_norm(Tensor input, Tensor? gamma, Tensor? beta,
                            Tensor? mean, Tensor? var,
                            bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor))SIG",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto input = args[0].ITensor(); // assumes non-static input Tensor
              auto orig_shape = input->getDimensions();
              auto shape = util::toVec(orig_shape);
              auto tensor_type = util::TRTDataTypeToScalarType(input->getType());
              auto options =
                  torch::TensorOptions().dtype(tensor_type).device(torch::kCUDA, ctx->settings.device.gpu_id);

              torch::Tensor gamma, beta, mean, var;
              LOG_DEBUG("Input :" << orig_shape << "/" << input->getType());
              // affine=True
              LOG_DEBUG("Args[1] gamma : " << args[1].isIValue() << " / " << args[1].IValue()->isNone());
              LOG_DEBUG("Args[2] beta : " << args[2].isIValue() << " / " << args[2].IValue()->isNone());
              // track_running_stats=True
              LOG_DEBUG("Args[3] mean : " << args[3].isIValue() << " / " << args[3].IValue()->isNone());
              LOG_DEBUG("Args[4] var : " << args[4].isIValue() << " / " << args[4].IValue()->isNone());
              LOG_DEBUG("use_input_stats, momemtum, cudnn_enabled disregarded");
              LOG_DEBUG("ctx->input_is_dynamic : " << ctx->input_is_dynamic);

              auto channel_dim = shape[1];
              if (ctx->input_is_dynamic) {
                gamma = args[1].unwrapToTensor(at::full(channel_dim, 1, options));
                beta = args[2].unwrapToTensor(at::full(channel_dim, 0, options));
                mean = args[3].unwrapToTensor();
                var = args[4].unwrapToTensor();
              } else {
                gamma = args[1].unwrapToTensor(at::full(channel_dim, 1, options));
                beta = args[2].unwrapToTensor(at::full(channel_dim, 0, options));
                mean = args[3].unwrapToTensor(at::full(channel_dim, 0, options));
                var = args[4].unwrapToTensor(at::full(channel_dim, 0, options));
              }

              auto eps = static_cast<float>(args[7].unwrapToDouble(1e-5f));

              TORCHTRT_CHECK(orig_shape.nbDims >= 2, "Unable to create batch normalization layer from node: " << *n);

              // Expand spatial dims from 1D to 2D if needed
              bool expandDims = (orig_shape.nbDims < 4);
              if (expandDims) {
                input = addPadding(ctx, n, input, 4);
              }

              _batch_norm(ctx, n, input, orig_shape, gamma, beta, mean, var, eps);

              return true;
            }})
        .pattern({
            R"SIG(aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias,
                              Tensor? running_mean, Tensor? running_var,
                              bool use_input_stats, float momentum, float eps,
                              bool cudnn_enabled) -> (Tensor))SIG",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto input = args[0].ITensorOrFreeze(ctx);
              auto orig_shape = input->getDimensions();
              auto shape = util::toVec(orig_shape);
              auto tensor_type = util::TRTDataTypeToScalarType(input->getType());
              auto options = torch::TensorOptions().dtype(tensor_type);

              LOG_DEBUG("Input :" << orig_shape << "/" << input->getType());
              // affine=True
              LOG_DEBUG("Args[1] weight : " << args[1].isIValue() << " / " << args[1].IValue()->isNone());
              LOG_DEBUG("Args[2] bias : " << args[2].isIValue() << " / " << args[2].IValue()->isNone());
              // track_running_stats=True
              LOG_DEBUG("Args[3] running_mean : " << args[3].isIValue() << " / " << args[3].IValue()->isNone());
              LOG_DEBUG("Args[4] running_var : " << args[4].isIValue() << " / " << args[4].IValue()->isNone());
              LOG_DEBUG("use_input_stats, momemtum, cudnn_enabled disregarded");
              LOG_DEBUG("ctx->input_is_dynamic : " << ctx->input_is_dynamic);

              // Expand spatial dims from 1D to 2D if needed
              bool expandDims = (orig_shape.nbDims < 4);
              if (expandDims) {
                input = addPadding(ctx, n, input, 4);
              }

              auto eps = static_cast<float>(args[7].unwrapToDouble(1e-5f));

              auto scales = args[1].unwrapToTensor(at::ones(shape[1], options)).cpu().contiguous();
              auto bias = args[2].unwrapToTensor(at::zeros(shape[1], options)).cpu().contiguous();

              // track_running_stats=True
              if (!args[3].IValue()->isNone() || !args[4].IValue()->isNone()) {
                auto running_mean = args[3].unwrapToTensor();
                auto running_var = args[4].unwrapToTensor();
                _batch_norm(
                    ctx,
                    n,
                    input,
                    orig_shape,
                    scales.to(running_mean.options()),
                    bias.to(running_mean.options()),
                    running_mean,
                    running_var,
                    eps);
                return true;
              }

              const int relu = 0;
              const float alpha = 0;
              LOG_DEBUG("Set parameter `relu` and `alpha` to 0");
              /*
              https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/namespacenvinfer1.html
              https://github.com/NVIDIA/TensorRT/tree/8.0.1/plugin/instanceNormalizationPlugin
              Type	      Parameter	  Description
              float	      epsilon	    A small number to prevent being divided by zero during normalization.
              Weights *	  scale	      A pointer to weights which contains information about scale factors for
                                      normalization. The definition of Weights can be found in the NvInfer.h header.
              Weights *	  bias        A pointer to weights which contains information about the bias values for
                                      normalization. The definition of Weights can be found in the NvInfer.h header.
              int	        relu	      A value used to enable leaky relu activation
              float	      alpha	      A small negative slope for the leaky relu activation
              */
              std::vector<nvinfer1::PluginField> f;
              f.emplace_back(nvinfer1::PluginField("epsilon", &eps, nvinfer1::PluginFieldType::kFLOAT32, 1));
              f.emplace_back(nvinfer1::PluginField(
                  "scales", scales.data_ptr<float>(), nvinfer1::PluginFieldType::kFLOAT32, scales.numel()));
              f.emplace_back(nvinfer1::PluginField(
                  "bias", bias.data_ptr<float>(), nvinfer1::PluginFieldType::kFLOAT32, bias.numel()));
              f.emplace_back(nvinfer1::PluginField("relu", &relu, nvinfer1::PluginFieldType::kINT32, 1));
              f.emplace_back(nvinfer1::PluginField("alpha", &alpha, nvinfer1::PluginFieldType::kFLOAT32, 1));

              nvinfer1::PluginFieldCollection fc;
              fc.nbFields = f.size();
              fc.fields = f.data();

              auto creator = getPluginRegistry()->getPluginCreator("InstanceNormalization_TRT", "1", "");
              auto instance_norm_plugin = creator->createPlugin("instance_norm", &fc);

              TORCHTRT_CHECK(
                  instance_norm_plugin, "Unable to create instance_norm plugin from TensorRT plugin registry" << *n);

              auto new_layer =
                  ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&input), 1, *instance_norm_plugin);

              new_layer->setName(util::node_info(n).c_str());
              auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));
              LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
              return true;
            }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
