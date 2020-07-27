#include <torch/torch.h>
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

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
    const std::string& name,
    float scalar = 1) {
  auto self_dims = self->getDimensions();
  auto self_dims_vec = util::toVec(self_dims);
  auto other_dims = other->getDimensions();
  auto other_dims_vec = util::toVec(other_dims);
  auto other_batch = other_dims_vec[0];

  // TODO: Proper broadcast check
  TRTORCH_CHECK(
      util::volume(self_dims) == util::volume(other_dims) ||
          util::volume(self_dims) == util::volume(other_dims) / other_batch,
      "Found inputs to elementwise operation do not have the same number of elements or is not broadcastable:\n  Found: self "
          << self_dims << " other " << other_dims);

  if (self_dims != other_dims) {
    LOG_DEBUG("Input shape dont match inserting shuffle layers to reshape to " << self_dims);
    auto self_shuffle = ctx->net->addShuffle(*self);
    self_shuffle->setReshapeDimensions(util::toDimsPad(self_dims_vec, other_dims_vec.size()));
    self_shuffle->setName(
        std::string("[Reshape self to " + util::toStr(self_dims) + " for broadcasting (" + name + ")]").c_str());
    self = self_shuffle->getOutput(0);
  }

  nvinfer1::ILayer* ele;
  if (scalar != 1) {
    LOG_WARNING("Please verify scalar handling in add converter, channel axis set to 3 but scaling is uniform");

    auto shape = util::toVec(other_dims);

    if (shape.size() < 4) {
      auto new_shape = util::toDimsPad(shape, 4);
      LOG_DEBUG(
          "Input shape is less than 4D got: "
          << util::toDims(shape) << ", inserting shuffle layers to reshape to 4D tensor shape: " << new_shape);
      auto other_shuffle = ctx->net->addShuffle(*other);
      other_shuffle->setReshapeDimensions(new_shape);
      other_shuffle->setName(std::string("[Reshape other to " + util::toStr(new_shape) + ']').c_str());
      other = other_shuffle->getOutput(0);

      auto self_shuffle = ctx->net->addShuffle(*self);
      self_shuffle->setReshapeDimensions(new_shape);
      self_shuffle->setName(std::string("[Reshape self to " + util::toStr(new_shape) + ']').c_str());
      self = self_shuffle->getOutput(0);
    }

    nvinfer1::ILayer* ele;
    if (scalar != 1) {
        LOG_WARNING("Please verify scalar handling in add converter, channel axis set to 3 but scaling is uniform");

        auto shape = util::toVec(other_dims);

        if (shape.size() < 4) {
             auto new_shape = util::toDimsPad(shape, 4);
             LOG_DEBUG("Input shape is less than 4D got: " << util::toDims(shape) << ", inserting shuffle layers to reshape to 4D tensor shape: " << new_shape);
             auto other_shuffle = ctx->net->addShuffle(*other);
             other_shuffle->setReshapeDimensions(new_shape);
             other_shuffle->setName(std::string("[Reshape other to " + util::toStr(new_shape) + ']').c_str());
             other = other_shuffle->getOutput(0);

             auto self_shuffle = ctx->net->addShuffle(*self);
             self_shuffle->setReshapeDimensions(new_shape);
             self_shuffle->setName(std::string("[Reshape self to " + util::toStr(new_shape) + ']').c_str());
             self = self_shuffle->getOutput(0);
        }

        auto tensor_type = util::toATenDType(self->getType());
        auto scale = Weights(ctx, scalar);
        auto power =  Weights(ctx, at::ones({1}).to(tensor_type));
        auto shift = Weights(ctx, at::zeros({1}).to(tensor_type));
        auto scaled = ctx->net->addScaleNd(*other, nvinfer1::ScaleMode::kUNIFORM, shift.data, scale.data, power.data, 0);
        auto scaled_other = scaled->getOutput(0);

        // if (shape.size() < 4) {
        //      LOG_DEBUG("Input shape wass less than 3D got, so was reshaped for scale, reshaping back to: " << util::toDims(shape));
        //      auto shuffle = ctx->net->addShuffle(*scaled_other);
        //      shuffle->setReshapeDimensions(util::toDims(shape));
        //      shuffle->setName(std::string("[Reshape other to " + util::toStr(util::toDims(shape)) + ']').c_str());
        //      scaled_other = shuffle->getOutput(0);
        // }

        ele = ctx->net->addElementWise(*self, *scaled_other, op);
    } else {
        ele = ctx->net->addElementWise(*self, *other, op);
    }

    return ele;

}

auto element_wise_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self + alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();
                    auto add = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n), scalar);

                    TRTORCH_CHECK(add, "Unable to create add layer from node: " << *n);

                    add->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self + alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();
                    auto add = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n), scalar);

                    TRTORCH_CHECK(add, "Unable to create add layer from node: " << *n);

                    add->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self - alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();
                    auto sub = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kSUB, self, other, util::node_info(n), scalar);

                    TRTORCH_CHECK(sub, "Unable to create sub layer from node: " << *n);

                    sub->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self / other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto div =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));

                    TRTORCH_CHECK(div, "Unable to create div layer from node: " << *n);

                    div->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // TODO: Remove with functionalization
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto div =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));

                    TRTORCH_CHECK(div, "Unable to create div layer from node: " << *n);

                    div->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto mul =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, other, util::node_info(n));

                    TRTORCH_CHECK(mul, "Unable to create mul layer from node: " << *n);

                    mul->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], mul->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // TODO: Remove with functionalization
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto mul =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, other, util::node_info(n));

                    TRTORCH_CHECK(mul, "Unable to create mul layer from node: " << *n);

                    mul->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], mul->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // TODO: Remove with functionalization
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto exponent = args[1].ITensorOrFreeze(ctx);
                    auto pow =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPOW, self, exponent, util::node_info(n));
                    TRTORCH_CHECK(pow, "Unable to create Power layer from node: " << *n);

                    pow->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], pow->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto exponentScalar = args[1].unwrapToScalar().to<float>();

                    // Calculate size of the input and define an exponent tensor of
                    // the same size
                    int volume = 1;
                    for (int i = 0; i < self->getDimensions().nbDims; i++) {
                      volume = volume * (self->getDimensions().d[i]);
                    }

                    // Create a torch tensor with constant exponent values
                    LOG_DEBUG("Broadcasting the exponent in power layer");
                    torch::Tensor exponentBlob = torch::full({volume}, exponentScalar);

                    // Create a corresponding constant layer in TRT and get the layer
                    // output.
                    auto weights = converters::Weights(ctx, exponentBlob);
                    auto exponentTensor = ctx->net->addConstant(self->getDimensions(), weights.data)->getOutput(0);

                    auto pow = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kPOW, self, exponentTensor, util::node_info(n));
                    TRTORCH_CHECK(pow, "Unable to create Power layer from node: " << *n);

                    pow->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], pow->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
