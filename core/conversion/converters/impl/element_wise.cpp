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
    // other is with dynamic shape, need to expand its dimension now and get its shape at runtime
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
      // size of dynamic dimension of other need to the same as that of corresponding dimension of self
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
      // other is with static shape, expand dimension to make tow tensor have the same number of dimension
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

auto element_wise_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> "
                  "Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self + alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();

                    if (1 != scalar) {
                      auto alphaTensor = tensor_to_const(ctx, torch::tensor({scalar}));
                      auto scaleLayer = add_elementwise(
                          ctx,
                          nvinfer1::ElementWiseOperation::kPROD,
                          other,
                          alphaTensor,
                          util::node_info(n) + std::string("_AlphaMultiplier"));
                      TRTORCH_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                      other = scaleLayer->getOutput(0);
                    }

                    auto add =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n));
                    TRTORCH_CHECK(add, "Unable to create add layer from node: " << *n);

                    add->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar "
                  "alpha=1) -> (Tensor(a!))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self + alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();

                    if (1 != scalar) {
                      auto alphaTensor = tensor_to_const(ctx, torch::tensor({scalar}));
                      auto scaleLayer = add_elementwise(
                          ctx,
                          nvinfer1::ElementWiseOperation::kPROD,
                          other,
                          alphaTensor,
                          util::node_info(n) + std::string("_AlphaMultiplier"));
                      TRTORCH_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                      other = scaleLayer->getOutput(0);
                    }

                    auto add =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n));
                    TRTORCH_CHECK(add, "Unable to create add layer from node: " << *n);

                    add->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self + alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[2].unwrapToScalar().to<float>() * args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));

                    auto add =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n));
                    TRTORCH_CHECK(add, "Unable to create add layer from node: " << *n);

                    add->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Compute min(max(min_threshold, input), max_threshold)
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto clamp_layer_out = self;
                    if (args[1].isIValue() && args[1].IValue()->isScalar()) {
                      auto minScalar = args[1].unwrapToScalar().to<float>();
                      auto minTensor = tensor_to_const(ctx, torch::tensor({minScalar}));
                      auto max_layer = add_elementwise(
                          ctx,
                          nvinfer1::ElementWiseOperation::kMAX,
                          clamp_layer_out,
                          minTensor,
                          util::node_info(n) + std::string("_max"));
                      TRTORCH_CHECK(max_layer, "Unable to create elementwise max layer for node: " << *n);
                      clamp_layer_out = max_layer->getOutput(0);
                    }

                    if (args[2].isIValue() && args[2].IValue()->isScalar()) {
                      auto maxScalar = args[2].unwrapToScalar().to<float>();
                      auto maxTensor = tensor_to_const(ctx, torch::tensor({maxScalar}));
                      auto min_layer = add_elementwise(
                          ctx,
                          nvinfer1::ElementWiseOperation::kMIN,
                          clamp_layer_out,
                          maxTensor,
                          util::node_info(n) + std::string("_min"));
                      TRTORCH_CHECK(min_layer, "Unable to create elementwise min layer for node: " << *n);
                      clamp_layer_out = min_layer->getOutput(0);
                    }

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], clamp_layer_out);
                    LOG_DEBUG("Clamp layer output tensor shape: " << clamp_layer_out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> "
                  "Tensor",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self - alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();
                    auto other = args[1].ITensorOrFreeze(ctx);

                    if (1 != scalar) {
                      auto scaleW = Weights(ctx, scalar);
                      auto unuse = Weights();
                      // IScaleLayer assert shift, scale and power to have
                      // the same dtype
                      auto scaleLayer = ctx->net->addScale(
                          *other, nvinfer1::ScaleMode::kUNIFORM, unuse.data, scaleW.data, unuse.data);
                      TRTORCH_CHECK(scaleLayer, "Unable to create scale layer from node: " << *n);
                      other = scaleLayer->getOutput(0);
                    }

                    auto sub =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, self, other, util::node_info(n));
                    TRTORCH_CHECK(sub, "Unable to create sub layer from node: " << *n);

                    sub->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar "
                  "alpha=1) -> (Tensor(a!))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // Should implement self - alpha * other
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto scalar = args[2].unwrapToScalar().to<float>();
                    auto other = args[1].ITensorOrFreeze(ctx);

                    if (1 != scalar) {
                      auto scaleW = Weights(ctx, scalar);
                      auto unuse = Weights();
                      // IScaleLayer assert shift, scale and power to have
                      // the same dtype
                      auto scaleLayer = ctx->net->addScale(
                          *other, nvinfer1::ScaleMode::kUNIFORM, unuse.data, scaleW.data, unuse.data);
                      TRTORCH_CHECK(scaleLayer, "Unable to create scale layer from node: " << *n);
                      other = scaleLayer->getOutput(0);
                    }

                    auto sub =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, self, other, util::node_info(n));
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
        .pattern({"aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
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
        .pattern({"aten::div_.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
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
        .pattern({"aten::ne.Tensor(Tensor self, Tensor other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto equal = add_elementwise(
                        ctx,
                        nvinfer1::ElementWiseOperation::kEQUAL,
                        self,
                        other,
                        util::node_info(n) + std::string("is_equal"));
                    TRTORCH_CHECK(equal, "Unable to create elementwise equal layer from node: " << *n);
                    // XOR with ones negates and produces not_equal result
                    auto options = torch::TensorOptions().dtype(torch::kFloat32);
                    auto ones = at::full({1}, 1, {options});
                    auto ones_tensor = tensor_to_const(ctx, ones);
                    nvinfer1::IIdentityLayer* cast_layer = ctx->net->addIdentity(*ones_tensor);
                    cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);

                    auto sub = add_elementwise(
                        ctx,
                        nvinfer1::ElementWiseOperation::kXOR,
                        cast_layer->getOutput(0),
                        equal->getOutput(0),
                        util::node_info(n));
                    TRTORCH_CHECK(sub, "Unable to create ne (not equal) layer from node: " << *n);

                    sub->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
                    LOG_DEBUG("Not equal layer output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::ne.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto scalar = args[1].unwrapToScalar().to<float>();
                    auto scalar_tensor = tensor_to_const(ctx, torch::tensor({scalar}));
                    auto equal = add_elementwise(
                        ctx,
                        nvinfer1::ElementWiseOperation::kEQUAL,
                        self,
                        scalar_tensor,
                        util::node_info(n) + std::string("is_equal"));
                    TRTORCH_CHECK(equal, "Unable to create elementwise equal layer from node: " << *n);
                    // XOR with ones negates and produces not_equal result
                    auto options = torch::TensorOptions().dtype(torch::kFloat32);
                    auto ones = at::full({1}, 1, {options});
                    auto ones_tensor = tensor_to_const(ctx, ones);
                    nvinfer1::IIdentityLayer* cast_layer = ctx->net->addIdentity(*ones_tensor);
                    cast_layer->setOutputType(0, nvinfer1::DataType::kBOOL);

                    auto sub = add_elementwise(
                        ctx,
                        nvinfer1::ElementWiseOperation::kXOR,
                        cast_layer->getOutput(0),
                        equal->getOutput(0),
                        util::node_info(n));
                    TRTORCH_CHECK(sub, "Unable to create ne (not equal) layer from node: " << *n);

                    sub->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
                    LOG_DEBUG("Not equal layer output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
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
                    auto exponent = tensor_to_const(ctx, torch::tensor({exponentScalar}));
                    auto pow =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPOW, self, exponent, util::node_info(n));
                    TRTORCH_CHECK(pow, "Unable to create Power layer from node: " << *n);

                    pow->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], pow->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::gt.Tensor(Tensor self, Tensor other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto gt =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n));
                    TRTORCH_CHECK(gt, "Unable to create greater layer from node: " << *n);

                    gt->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gt->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::gt.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
                    auto gt =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n));
                    TRTORCH_CHECK(gt, "Unable to create greater layer from node: " << *n);

                    gt->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gt->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::lt.Tensor(Tensor self, Tensor other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto lt =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n));
                    TRTORCH_CHECK(lt, "Unable to create less layer from node: " << *n);

                    lt->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], lt->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::lt.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
                    auto lt =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n));
                    TRTORCH_CHECK(lt, "Unable to create less layer from node: " << *n);

                    lt->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], lt->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::eq.Tensor(Tensor self, Tensor other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);
                    auto eq =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n));
                    TRTORCH_CHECK(eq, "Unable to create equal layer from node: " << *n);

                    eq->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], eq->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::eq.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
                    auto eq =
                        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n));
                    TRTORCH_CHECK(eq, "Unable to create equal layer from node: " << *n);

                    eq->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], eq->getOutput(0));
                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::ge.Tensor(Tensor self, Tensor other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);

                    auto greater = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n) + "_greater");
                    TRTORCH_CHECK(greater, "Unable to create Greater layer from node: " << *n);

                    auto equal = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
                    TRTORCH_CHECK(equal, "Unable to create Equal layer from node: " << *n);

                    auto or_op = ctx->net->addElementWise(
                        *greater->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

                    TRTORCH_CHECK(or_op, "Unable to create Or layer from node: " << *n);
                    or_op->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::ge.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));

                    auto greater = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n) + "_greater");
                    TRTORCH_CHECK(greater, "Unable to create Greater layer from node: " << *n);

                    auto equal = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
                    TRTORCH_CHECK(equal, "Unable to create Equal layer from node: " << *n);

                    auto or_op = ctx->net->addElementWise(
                        *greater->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

                    TRTORCH_CHECK(or_op, "Unable to create Or layer from node: " << *n);
                    or_op->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::le.Tensor(Tensor self, Tensor other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto other = args[1].ITensorOrFreeze(ctx);

                    auto less = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n) + "_less");
                    TRTORCH_CHECK(less, "Unable to create Less layer from node: " << *n);

                    auto equal = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
                    TRTORCH_CHECK(equal, "Unable to create Equal layer from node: " << *n);

                    auto or_op = ctx->net->addElementWise(
                        *less->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

                    TRTORCH_CHECK(or_op, "Unable to create Or layer from node: " << *n);
                    or_op->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }})
        .pattern({"aten::le.Scalar(Tensor self, Scalar other) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto otherScalar = args[1].unwrapToScalar().to<float>();
                    auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));

                    auto less = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n) + "_less");
                    TRTORCH_CHECK(less, "Unable to create Less layer from node: " << *n);

                    auto equal = add_elementwise(
                        ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
                    TRTORCH_CHECK(equal, "Unable to create Equal layer from node: " << *n);

                    auto or_op = ctx->net->addElementWise(
                        *less->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

                    TRTORCH_CHECK(or_op, "Unable to create Or layer from node: " << *n);
                    or_op->setName(util::node_info(n).c_str());
                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());
                    return true;
                  }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
