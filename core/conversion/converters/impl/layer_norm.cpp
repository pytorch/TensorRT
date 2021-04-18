#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

nvinfer1::ILayer* add_elementwise(
    ConversionCtx* ctx,
    nvinfer1::ElementWiseOperation op,
    nvinfer1::ITensor* self,
    nvinfer1::ITensor* other,
    const std::string& name) {
  // ensure self to have larger number of dimension
  bool swapSelfOther = false;
  if (self->getDimensions().nbDims < other->getDimensions().nbDims) {
    std::swap(self, other);
    swapSelfOther = true;
  }
  auto selfDim = util::toVec(self->getDimensions());
  auto otherDim = util::toVec(other->getDimensions());
  if (selfDim.size() != otherDim.size()) {
    // other is with dynamic shape, need to expand its dimension now and get its
    // shape at runtime
    if (otherDim.end() != std::find(otherDim.begin(), otherDim.end(), -1)) {
      auto thOtherStaticShapeMask = torch::ones(selfDim.size(), torch::kInt32);
      auto thOtherDynamicShapeMask = torch::zeros(selfDim.size(), torch::kInt32);
      for (size_t start = selfDim.size() - otherDim.size(), idx = 0; idx < otherDim.size(); ++idx) {
        if (-1 != otherDim[idx]) {
          thOtherStaticShapeMask[start + idx] = otherDim[idx];
        } else {
          thOtherStaticShapeMask[start + idx] = 0;
          thOtherDynamicShapeMask[start + idx] = 1;
        }
      }
      auto otherStaticShapeMask = tensor_to_const(ctx, thOtherStaticShapeMask);
      auto otherDynamicShapeMask = tensor_to_const(ctx, thOtherDynamicShapeMask);
      auto selfShape = ctx->net->addShape(*self)->getOutput(0);
      // size of dynamic dimension of other need to the same as that of
      // corresponding dimension of self
      auto otherDynamicShape =
          ctx->net->addElementWise(*selfShape, *otherDynamicShapeMask, nvinfer1::ElementWiseOperation::kPROD)
              ->getOutput(0);
      auto targetOtherShape =
          ctx->net->addElementWise(*otherDynamicShape, *otherStaticShapeMask, nvinfer1::ElementWiseOperation::kSUM)
              ->getOutput(0);

      auto otherShuffle = ctx->net->addShuffle(*other);
      otherShuffle->setName(std::string("Reshape other tensor to have the same nDim as self for " + name).c_str());
      otherShuffle->setInput(1, *targetOtherShape);
      other = otherShuffle->getOutput(0);
    } else {
      // other is with static shape, expand dimension to make tow tensor have
      // the same number of dimension
      auto otherShuffle = ctx->net->addShuffle(*other);
      otherShuffle->setReshapeDimensions(util::toDimsPad(otherDim, selfDim.size()));
      other = otherShuffle->getOutput(0);
    }
  }
  if (swapSelfOther) {
    // swap back
    std::swap(self, other);
    swapSelfOther = false;
  }
  auto ele = ctx->net->addElementWise(*self, *other, op);
  ele->setName(name.c_str());
  return ele;
}

auto layer_norm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern({
    R"SIG(aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? gamma, Tensor? beta,
                           float eps, bool cudnn_enabled) -> (Tensor))SIG",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      auto input = args[0].ITensor(); // assumes non-static input Tensor
      auto orig_shape = input->getDimensions();
      auto shape = util::toVec(orig_shape);

      /* Layer_Norm normalizes over last N dimensions.
         normalizaed_shape could be (C,H,W), (H,W), or (W). */

      auto normalized_shape = args[1].unwrapToIntList();
      auto normalized_shape_vec = util::toVec(util::toDims(normalized_shape));

      torch::Tensor gamma, beta;
      gamma = args[2].unwrapToTensor();
      beta = args[3].unwrapToTensor();

      // Remove batch dimension from input shape for expand_size, which will
      // be used to create weights for addScaleNd later.
      auto expand_size = shape;
      expand_size.erase(expand_size.begin(), expand_size.begin() + 1);
      auto gamma_expand = gamma.expand(expand_size);
      auto beta_expand = beta.expand(expand_size);

      // Unwrap eps.
      auto eps = args[4].unwrapToDouble();
      LOG_DEBUG("cudnn disregarded");

      // Set up  axis_ask for E[x].
      uint32_t axis_mask = 0;
      for (size_t i = 0; i < normalized_shape_vec.size(); i++) {
        axis_mask |= 1 << (shape.size() - i - 1);
      }
      LOG_DEBUG("Axis Mask for E[x]" << std::bitset<32>(axis_mask));

      // E[x]
      auto mean_layer_expected = ctx->net->addReduce(*input, nvinfer1::ReduceOperation::kAVG, axis_mask, false);
      TRTORCH_CHECK(mean_layer_expected, "Unable to create mean_layer_expected from node: " << *n);
      mean_layer_expected->setName((util::node_info(n) + "_mean_expected").c_str());
      auto mean_layer_expected_out = mean_layer_expected->getOutput(0);

      // Expand output of E[x] to the same shape as original input.
      c10::List<int64_t> repeats_expected;
      for (size_t i = 0; i < shape.size(); i++) {
        auto repeat = i > (shape.size() - normalized_shape_vec.size() - 1) ? shape[i] : 1;
        repeats_expected.push_back(repeat);
      }

      int repeats_expected_rank = repeats_expected.size();
      auto mean_layer_expected_out_dims = mean_layer_expected_out->getDimensions();
      auto num_expand_dims_expected = repeats_expected_rank - mean_layer_expected_out_dims.nbDims;

      if (num_expand_dims_expected > 0) {
        nvinfer1::Dims reshape_expected_dims;
        reshape_expected_dims.nbDims = repeats_expected.size();
        for (int i = 0; i < num_expand_dims_expected; i++) {
          reshape_expected_dims.d[repeats_expected.size() - 1 - i] = 1;
        }
        for (int i = 0; i < mean_layer_expected_out_dims.nbDims; i++) {
          reshape_expected_dims.d[i] = mean_layer_expected_out_dims.d[i];
        }
        // Add a reshape layer to expand dims
        auto reshape_layer_expected = ctx->net->addShuffle(*mean_layer_expected_out);
        reshape_layer_expected->setReshapeDimensions(reshape_expected_dims);
        mean_layer_expected_out = reshape_layer_expected->getOutput(0);
      }

      for (int i = repeats_expected.size() - 1; i >= 0; --i) {
        std::vector<nvinfer1::ITensor*> tensors_vec;
        for (int j = 0; j < repeats_expected[i]; j++) {
          tensors_vec.push_back(mean_layer_expected_out);
        }
        auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
        concat_layer->setAxis(i);
        mean_layer_expected_out = concat_layer->getOutput(0);
      }

      // X-E[x]
      auto sub = add_elementwise(
          ctx,
          nvinfer1::ElementWiseOperation::kSUB,
          input,
          mean_layer_expected_out,
          (util::node_info(n) + "_sub").c_str());
      TRTORCH_CHECK(sub, "Unable to create Add layer from node: " << *n);
      sub->setName((util::node_info(n) + "_sub").c_str());
      auto xsubmean = sub->getOutput(0);

      // Variance
      float pow_scalar = 2;
      auto exponent = tensor_to_const(ctx, torch::tensor({pow_scalar}));
      auto pow = add_elementwise(
          ctx, nvinfer1::ElementWiseOperation::kPOW, xsubmean, exponent, (util::node_info(n) + "_pow").c_str());
      TRTORCH_CHECK(pow, "Unable to create Power layer from node: " << *n);
      pow->setName((util::node_info(n) + "_pow").c_str());
      auto pow_out = pow->getOutput(0);

      auto mean_layer_var = ctx->net->addReduce(*pow_out, nvinfer1::ReduceOperation::kAVG, axis_mask, false);
      TRTORCH_CHECK(mean_layer_var, "Unable to create mean_layer_var from node: " << *n);
      mean_layer_var->setName((util::node_info(n) + "_mean_var").c_str());
      auto mean_layer_var_out = mean_layer_var->getOutput(0);

      // Expand output of mean_layer_var to the same shape as original
      // input.
      c10::List<int64_t> repeats_var;
      for (size_t i = 0; i < shape.size(); i++) {
        auto repeat = i > (shape.size() - normalized_shape_vec.size() - 1) ? shape[i] : 1;
        repeats_var.push_back(repeat);
      }

      int repeats_var_rank = repeats_var.size();
      auto mean_layer_var_out_dims = mean_layer_var_out->getDimensions();
      auto num_expand_dims_var = repeats_var_rank - mean_layer_var_out_dims.nbDims;

      if (num_expand_dims_var > 0) {
        nvinfer1::Dims reshape_dims_var;
        reshape_dims_var.nbDims = repeats_var.size();
        for (int i = 0; i < num_expand_dims_var; i++) {
          reshape_dims_var.d[repeats_var.size() - 1 - i] = 1;
        }
        for (int i = 0; i < mean_layer_var_out_dims.nbDims; i++) {
          reshape_dims_var.d[i] = mean_layer_var_out_dims.d[i];
        }

        // Add a reshape layer to expand dims
        auto reshape_layer_var = ctx->net->addShuffle(*mean_layer_var_out);
        reshape_layer_var->setReshapeDimensions(reshape_dims_var);
        mean_layer_var_out = reshape_layer_var->getOutput(0);
      }

      for (int i = repeats_var.size() - 1; i >= 0; --i) {
        std::vector<nvinfer1::ITensor*> tensors_vec;
        for (int j = 0; j < repeats_var[i]; j++) {
          tensors_vec.push_back(mean_layer_var_out);
        }
        auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
        concat_layer->setAxis(i);
        mean_layer_var_out = concat_layer->getOutput(0);
      }

      // add eps
      auto eps_tensor = tensor_to_const(ctx, torch::tensor({eps}));
      auto add = add_elementwise(
          ctx,
          nvinfer1::ElementWiseOperation::kSUM,
          mean_layer_var_out,
          eps_tensor,
          (util::node_info(n) + "_add").c_str());
      TRTORCH_CHECK(add, "Unable to create Add layer from node: " << *n);
      add->setName((util::node_info(n) + "_add").c_str());
      auto add_out = add->getOutput(0);

      // add Unary layer for sqrt((var + eps))
      auto unary = ctx->net->addUnary(*add_out, nvinfer1::UnaryOperation::kSQRT);
      TRTORCH_CHECK(unary, "Unable to create unary layer from node: " << *n);
      unary->setName((util::node_info(n) + "_unary_sqrt").c_str());
      auto unary_out = unary->getOutput(0);

      // (x - E[x]) / sqrt((var + eps))
      auto div = add_elementwise(
          ctx, nvinfer1::ElementWiseOperation::kDIV, xsubmean, unary_out, (util::node_info(n) + "_div").c_str());
      TRTORCH_CHECK(div, "Unable to create div layer from node: " << *n);
      div->setName((util::node_info(n) + "_div").c_str());
      auto div_out = div->getOutput(0);

      // Set up gamma_weights and beta_weights from gamma_expand and
      // beta_expand
      auto gamma_weights = Weights(ctx, gamma_expand);
      auto beta_weights = Weights(ctx, beta_expand);

      auto power = Weights(ctx, at::ones_like(gamma_expand));
      auto scale_nd = ctx->net->addScaleNd(
          *div_out, nvinfer1::ScaleMode::kELEMENTWISE, beta_weights.data, gamma_weights.data, power.data, 1);

      scale_nd->setName((util::node_info(n) + "_scale_nd").c_str());
      auto scale_nd_out = scale_nd->getOutput(0);

      ctx->AssociateValueAndTensor(n->outputs()[0], scale_nd_out);
      return true;
    }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
