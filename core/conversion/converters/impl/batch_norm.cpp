#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
// #include "core/plugins/plugin_prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

static inline at::Tensor repeat_if_defined(const at::Tensor& t, int64_t repeat) {
    if (t.defined()) {
        return t.repeat(repeat);
    }
    return t;
}

auto batch_norm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::batch_norm(Tensor input, Tensor? gamma, Tensor? beta,
                            Tensor? mean, Tensor? var,
                            bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);
      auto tensor_type = util::toATenDType(input->getType());
      auto options = torch::TensorOptions().dtype(tensor_type);

      torch::Tensor gamma, beta, mean, var;

      if (ctx->input_is_dynamic) {
        gamma = args[1].unwrapToTensor();
        beta = args[2].unwrapToTensor();
        mean = args[3].unwrapToTensor();
        var = args[4].unwrapToTensor();
      } else {
        gamma = args[1].unwrapToTensor(at::full({shape}, 1, {options}));
        beta = args[2].unwrapToTensor(at::full({shape}, 1, {options}));
        mean = args[3].unwrapToTensor(at::full({shape}, 0, {options}));
        var = args[4].unwrapToTensor(at::full({shape}, 0, {options}));
      }

      auto eps = args[7].unwrapToDouble(1e-5f);

      LOG_DEBUG("momentum disregarded");
      LOG_DEBUG("training disregarded");
      LOG_DEBUG("cudnn disregarded");

      auto should_unpack = util::toVec(orig_shape).size() < 4;
      if (should_unpack) {
        // expand spatial dims from 1D to 2D
        auto new_shape = util::toDimsTailPad(util::toVec(orig_shape), 4);
        LOG_DEBUG(
            "Input shape is less than 4D got: "
            << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
        auto in_shuffle = ctx->net->addShuffle(*input);
        in_shuffle->setReshapeDimensions(new_shape);
        in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
        input = in_shuffle->getOutput(0);
      }

      auto scale = gamma / torch::sqrt(var + eps);
      auto bias = beta - mean * scale;

      auto scale_weights = Weights(ctx, scale);
      auto bias_weights = Weights(ctx, bias);

      auto power = Weights(ctx, at::ones_like(scale));
      auto bn = ctx->net->addScaleNd(
          *input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
      bn->setName(util::node_info(n).c_str());
      auto out_tensor = bn->getOutput(0);

      if (should_unpack) {
        LOG_DEBUG("Inserting shuffle layer to reshape to back to original shape: " << orig_shape);
        auto out_shuffle = ctx->net->addShuffle(*out_tensor);
        out_shuffle->setReshapeDimensions(orig_shape);
        out_shuffle->setName(std::string("[Reshape output to " + util::toStr(orig_shape) + ']').c_str());
        out_tensor = out_shuffle->getOutput(0);
      }

      ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
      return true;
    }}).pattern({
     R"SIG(aten::instance_norm(Tensor input, Tensor? gamma, Tensor? beta,
           Tensor? mean, Tensor? var,
           bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> (Tensor))SIG",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
         auto input = args[0].ITensorOrFreeze(ctx); // assumes non-static input Tensor
         auto orig_shape = input->getDimensions();
         auto shape = util::toVec(orig_shape);
         auto tensor_type = util::toATenDType(input->getType());
         LOG_DEBUG("input type is: "<< tensor_type);
         auto options = torch::TensorOptions().dtype(tensor_type);

        if(args[3].IValue()->isNone() && args[4].IValue()->isNone())
        {
            // Use tensorrt plugin layer when running mean and running variance are not set.
            Weights gamma, beta;
         if (ctx->input_is_dynamic) {
             gamma = Weights(ctx, args[1].unwrapToTensor());
             beta = Weights(ctx, args[2].unwrapToTensor());
         } else {
             gamma = Weights(ctx, args[1].unwrapToTensor(at::full({shape}, 1, {options})));
             beta = Weights(ctx, args[2].unwrapToTensor(at::full({shape}, 1, {options})));
         }

         LOG_DEBUG("gamma size: "<< gamma.data.count);
         LOG_DEBUG("beta size: "<< beta.data.count);

         auto epsilon = args[7].unwrapToDouble(1e-5f);

         LOG_DEBUG("momentum disregarded");
         LOG_DEBUG("use_input_stats disregarded");
         LOG_DEBUG("cudnn disregarded");

         auto should_unpack = util::toVec(orig_shape).size() < 4;
         if (should_unpack) {
             // expand spatial dims from 1D to 2D
             auto new_shape = util::toDimsPad(util::toVec(orig_shape), 4);
             LOG_DEBUG(
                     "Input shape is less than 4D got: "
                             << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
             auto in_shuffle = ctx->net->addShuffle(*input);
             in_shuffle->setReshapeDimensions(new_shape);
             in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
             input = in_shuffle->getOutput(0);
         }
         std::string pluginName = "InstanceNormalization_TRT";
         nvinfer1::PluginFieldCollection fc;
         std::vector<nvinfer1::PluginField> f;
         f.emplace_back(nvinfer1::PluginField("epsilon", &epsilon, nvinfer1::PluginFieldType::kFLOAT32, 1));
         f.emplace_back(nvinfer1::PluginField("scales", gamma.data.values, nvinfer1::PluginFieldType::kFLOAT32, gamma.data.count));
         f.emplace_back(nvinfer1::PluginField("bias", beta.data.values, nvinfer1::PluginFieldType::kFLOAT32, beta.data.count));

         fc.nbFields = f.size();
         fc.fields = f.data();
         LOG_DEBUG("=========IN PLUGIN INSTANCE NORM===========");
         // nvinfer1::IPluginV2* pluginV2 = ctx->mPluginRegistry.at(pluginName)->createPlugin("instancenorm", &fc);
         // std::unordered_map<std::string, nvinfer1::IPluginCreator*> mPluginRegistry = plugins::registerPlugins();

         nvinfer1::IPluginV2* pluginV2 = getPluginRegistry()->getPluginCreator("NormalizePlugin", "1", "trtorch")->createPlugin("instancenorm", &fc);

         TRTORCH_CHECK(pluginV2, "Unable to create interpolation plugin from node" << *n);
         auto instance_norm_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&input), 1, *pluginV2);
         TRTORCH_CHECK(instance_norm_layer, "Unable to create instance norm plugin from node" << *n);

         instance_norm_layer->setName("instancenorm");
         auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], instance_norm_layer->getOutput(0));
         LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
         return true;
         }
         else
         {
             LOG_DEBUG("=========IN NORMAL INSTANCE NORM===========");
             torch::Tensor gamma, beta, mean, var;

             if (ctx->input_is_dynamic) {
                 gamma = args[1].unwrapToTensor();
                 beta = args[2].unwrapToTensor();
                 mean = args[3].unwrapToTensor();
                 var = args[4].unwrapToTensor();
             } else {
                 gamma = args[1].unwrapToTensor(at::full({shape}, 1, {options}));
                 beta = args[2].unwrapToTensor(at::full({shape}, 1, {options}));
                 mean = args[3].unwrapToTensor(at::full({shape}, 0, {options}));
                 var = args[4].unwrapToTensor(at::full({shape}, 0, {options}));
             }

             auto eps = args[7].unwrapToDouble(1e-5f);

             LOG_DEBUG("momentum disregarded");
             LOG_DEBUG("training disregarded");
             LOG_DEBUG("cudnn disregarded");

             auto should_unpack = util::toVec(orig_shape).size() < 4;
             if (should_unpack) {
                 // expand spatial dims from 1D to 2D
                 auto new_shape = util::toDimsPad(util::toVec(orig_shape), 4);
                 LOG_DEBUG(
                         "Input shape is less than 4D got: "
                                 << orig_shape << ", inserting shuffle layer to reshape to 4D tensor shape: " << new_shape);
                 auto in_shuffle = ctx->net->addShuffle(*input);
                 in_shuffle->setReshapeDimensions(new_shape);
                 in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape) + ']').c_str());
                 input = in_shuffle->getOutput(0);
             }

             auto new_shape = util::toVec(input->getDimensions());
             int64_t b = new_shape[0];
             int64_t c = new_shape[1];
             new_shape[1] = b * c;
             new_shape[0] = 1;

             at::Tensor gamma_ = repeat_if_defined(gamma, b);
             at::Tensor beta_ = repeat_if_defined(beta, b);
             at::Tensor mean_ = repeat_if_defined(mean, b);
             at::Tensor var_ = repeat_if_defined(var, b);

             auto in_shuffle = ctx->net->addShuffle(*input);
             auto new_shape_dim = util::toDimsPad(new_shape, 4);
             in_shuffle->setReshapeDimensions(new_shape_dim);
             in_shuffle->setName(std::string("[Reshape input to " + util::toStr(new_shape_dim) + ']').c_str());
             input = in_shuffle->getOutput(0);

             auto scale = gamma_ / torch::sqrt(var_ + eps);
             auto bias = beta_ - mean_ * scale;

             auto scale_weights = Weights(ctx, scale);
             auto bias_weights = Weights(ctx, bias);

             auto power = Weights(ctx, at::ones_like(scale));
             auto bn = ctx->net->addScaleNd(
                     *input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
             bn->setName(util::node_info(n).c_str());
             auto out_tensor = bn->getOutput(0);

             if (should_unpack) {
                 LOG_DEBUG("Inserting shuffle layer to reshape to back to original shape: " << orig_shape);
                 auto out_shuffle = ctx->net->addShuffle(*out_tensor);
                 out_shuffle->setReshapeDimensions(orig_shape);
                 out_shuffle->setName(std::string("[Reshape output to " + util::toStr(orig_shape) + ']').c_str());
                 out_tensor = out_shuffle->getOutput(0);
             }

             ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
             return true;
         }
     }});;

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
