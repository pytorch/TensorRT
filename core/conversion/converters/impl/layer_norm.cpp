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

nvinfer1::ITensor* broadcast(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* to_broadcast,
    const int nbDims,
    const std::string& tag) {
  auto to_broadcast_nbdims = to_broadcast->getDimensions().nbDims;
  TORCHTRT_CHECK(to_broadcast_nbdims <= nbDims, "Cannot broadcast tensor with more dimensions than the target");
  if (to_broadcast_nbdims == nbDims) {
    return to_broadcast;
  }
  auto shape_layer = ctx->net->addShape(*to_broadcast);
  TORCHTRT_CHECK(shape_layer, "Unable to create shape layer from node: " << *n);
  shape_layer->setName((util::node_info(n) + "_shape_" + tag).c_str());
  auto shape_layer_out = shape_layer->getOutput(0);

  auto extra_dims_tensor = torch::ones({nbDims - to_broadcast_nbdims}, torch::TensorOptions().dtype(torch::kInt32));
  auto extra_dims_itensor = tensor_to_const(ctx, extra_dims_tensor);

  std::vector<nvinfer1::ITensor*> to_concat = {extra_dims_itensor, shape_layer_out};
  auto concat_layer = ctx->net->addConcatenation(to_concat.data(), to_concat.size());
  TORCHTRT_CHECK(concat_layer, "Unable to create concat layer from node: " << *n);
  concat_layer->setName((util::node_info(n) + "_concat_" + tag).c_str());
  auto target_shape = concat_layer->getOutput(0);

  auto shuffle_layer = ctx->net->addShuffle(*to_broadcast);
  TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
  shuffle_layer->setName((util::node_info(n) + "_shuffle_" + tag).c_str());
  shuffle_layer->setInput(1, *target_shape);
  auto output = shuffle_layer->getOutput(0);
  LOG_DEBUG(
      "Broadcast " << tag << " to shape: " << output->getDimensions() << " from " << to_broadcast->getDimensions());
  return output;
}

auto layer_norm_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? gamma, Tensor? beta,
                           float eps, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensorOrFreeze(ctx);
      auto input_shape = input->getDimensions();
      auto input_shape_vec = util::toVec(input_shape);
      auto normalized_shape = args[1].unwrapToIntList();
      auto normalized_shape_vec = util::toVec(util::toDims(normalized_shape));
      auto axis = input_shape_vec.size() - normalized_shape_vec.size();
      uint32_t axes_mask = 0;
      for (size_t i = axis; i < input_shape_vec.size(); i++) {
        axes_mask |= 1 << i;
      }

      nvinfer1::ITensor* gamma = nullptr;
      if (args[2].IValue()->isNone()) {
        auto gamma_torch_tensor = torch::ones(input_shape_vec, torch::TensorOptions().dtype(torch::kFloat32));
        gamma = tensor_to_const(ctx, gamma_torch_tensor);
      } else {
        gamma = args[2].ITensorOrFreeze(ctx);
        gamma = broadcast(ctx, n, gamma, input_shape_vec.size(), "gamma");
      }

      nvinfer1::ITensor* beta = nullptr;
      if (args[3].IValue()->isNone()) {
        auto beta_torch_tensor = torch::zeros(input_shape_vec, torch::TensorOptions().dtype(torch::kFloat32));
        beta = tensor_to_const(ctx, beta_torch_tensor);
      } else {
        beta = args[3].ITensorOrFreeze(ctx);
        beta = broadcast(ctx, n, beta, input_shape_vec.size(), "beta");
      }

      auto eps = args[4].unwrapToDouble();

      auto normalize_layer = ctx->net->addNormalization(*input, *gamma, *beta, axes_mask);
      TORCHTRT_CHECK(normalize_layer, "Unable to create layer_norm from node: " << *n);
      normalize_layer->setName(util::node_info(n).c_str());
      normalize_layer->setEpsilon(eps);
      normalize_layer->setComputePrecision(nvinfer1::DataType::kFLOAT);
      auto normalized = normalize_layer->getOutput(0);

      ctx->AssociateValueAndTensor(n->outputs()[0], normalized);
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
