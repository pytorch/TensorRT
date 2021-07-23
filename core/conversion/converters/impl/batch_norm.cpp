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
      TRTORCH_CHECK(orig_shape.nbDims > 2 , "Unable to create batch normalization layer from node: " << *n);

      // Expand spatial dims from 1D to 2D if needed
      bool expandDims = (orig_shape.nbDims < 4);

      if (expandDims) {
        input = addPadding(ctx, n, input, 4);
      }

      auto scale = gamma / torch::sqrt(var + eps);
      auto bias = beta - mean * scale;

      auto scale_weights = Weights(ctx, scale);
      auto bias_weights = Weights(ctx, bias);

      auto power = Weights(ctx, at::ones_like(scale));
      auto bn = ctx->net->addScaleNd(
          *input, nvinfer1::ScaleMode::kCHANNEL, bias_weights.data, scale_weights.data, power.data, 1);
      bn->setName(util::node_info(n).c_str());
      // Un-pad bn output if needed
      auto out_tensor = addUnpadding(ctx, n, bn->getOutput(0), orig_shape.nbDims);
      ctx->AssociateValueAndTensor(n->outputs()[0], out_tensor);
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
