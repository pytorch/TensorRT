#include "c10/util/MathConstants.h"
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

nvinfer1::ITensor* clamp_util(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* self,
    float limit,
    nvinfer1::ElementWiseOperation op_type,
    std::string str) {
  nvinfer1::ITensor* clamp_layer_out = self;
  auto limitTensor = tensor_to_const(ctx, torch::tensor({limit}));
  auto limit_layer = add_elementwise(ctx, op_type, clamp_layer_out, limitTensor, util::node_info(n) + str);
  TORCHTRT_CHECK(limit_layer, "Unable to create elementwise " << str << " layer for node: " << *n);
  clamp_layer_out = limit_layer->getOutput(0);
  return clamp_layer_out;
}

auto element_wise_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> "
             "Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self + alpha * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto scalar = args[2].unwrapToScalar();

               if (1 != scalar.to<float>()) {
                 auto alphaTensor = scalar_to_tensor(ctx, scalar);
                 auto scaleLayer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     other,
                     alphaTensor,
                     util::node_info(n) + std::string("_AlphaMultiplier"));
                 TORCHTRT_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                 other = scaleLayer->getOutput(0);
               }

               auto add = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n));
               TORCHTRT_CHECK(add, "Unable to create add layer from node: " << *n);

               add->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar "
             "alpha=1) -> (Tensor(a!))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self + alpha * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto scalar = args[2].unwrapToScalar();

               if (1 != scalar.to<float>()) {
                 auto alphaTensor = scalar_to_tensor(ctx, scalar);
                 auto scaleLayer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     other,
                     alphaTensor,
                     util::node_info(n) + std::string("_AlphaMultiplier"));
                 TORCHTRT_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                 other = scaleLayer->getOutput(0);
               }

               auto add = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n));
               TORCHTRT_CHECK(add, "Unable to create add layer from node: " << *n);

               add->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self + alpha * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto otherScalar = args[2].unwrapToScalar().to<float>() * args[1].unwrapToScalar().to<float>();
               auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));

               auto add = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, util::node_info(n));
               TORCHTRT_CHECK(add, "Unable to create add layer from node: " << *n);

               add->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], add->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Compute min(max(min_threshold, input), max_threshold)
               auto self = args[0].ITensorOrFreeze(ctx);
               auto clamp_layer_out = self;

               if (args[1].isIValue() && args[1].IValue()->isScalar() && args[2].isIValue() &&
                   args[2].IValue()->isScalar()) {
                 auto alpha = args[1].unwrapToScalar().to<float>();
                 auto beta = args[2].unwrapToScalar().to<float>();
                 auto clip_layer = ctx->net->addActivation(*self, nvinfer1::ActivationType::kCLIP);
                 TORCHTRT_CHECK(clip_layer, "Unable to create clip layer for node: " << *n);
                 clip_layer->setAlpha(alpha);
                 clip_layer->setBeta(beta);
                 clamp_layer_out = clip_layer->getOutput(0);
               } else if (args[1].isIValue() && args[1].IValue()->isScalar()) {
                 auto limit = args[1].unwrapToScalar().to<float>();
                 clamp_layer_out = clamp_util(ctx, n, self, limit, nvinfer1::ElementWiseOperation::kMAX, "_max");
               } else if (args[2].isIValue() && args[2].IValue()->isScalar()) {
                 auto limit = args[2].unwrapToScalar().to<float>();
                 clamp_layer_out = clamp_util(ctx, n, self, limit, nvinfer1::ElementWiseOperation::kMIN, "_min");
               }

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], clamp_layer_out);
               LOG_DEBUG("Clamp layer output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::clamp_min(Tensor self, Scalar min) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Compute min(max(min_threshold, input), max_threshold)
               auto self = args[0].ITensorOrFreeze(ctx);
               auto clamp_layer_out = self;
               if (args[1].isIValue() && args[1].IValue()->isScalar()) {
                 auto limit = args[1].unwrapToScalar().to<float>();
                 clamp_layer_out = clamp_util(ctx, n, self, limit, nvinfer1::ElementWiseOperation::kMAX, "_max");
               }

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], clamp_layer_out);
               LOG_DEBUG("clamp_min layer output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::clamp_max(Tensor self, Scalar max) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Compute min(max(min_threshold, input), max_threshold)
               auto self = args[0].ITensorOrFreeze(ctx);
               auto clamp_layer_out = self;
               if (args[1].isIValue() && args[1].IValue()->isScalar()) {
                 auto limit = args[1].unwrapToScalar().to<float>();
                 clamp_layer_out = clamp_util(ctx, n, self, limit, nvinfer1::ElementWiseOperation::kMIN, "_min");
               }

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], clamp_layer_out);
               LOG_DEBUG("clamp_max layer output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> "
             "Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self - alpha * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto scalar = args[2].unwrapToScalar();

               if (1 != scalar.to<float>()) {
                 auto alphaTensor = scalar_to_tensor(ctx, scalar);
                 auto scaleLayer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     other,
                     alphaTensor,
                     util::node_info(n) + std::string("_AlphaMultiplier"));
                 TORCHTRT_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                 other = scaleLayer->getOutput(0);
               }

               auto sub = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, self, other, util::node_info(n));
               TORCHTRT_CHECK(sub, "Unable to create sub layer from node: " << *n);

               sub->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self - alpha * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].unwrapToScalar().to<float>();
               auto alpha = args[2].unwrapToScalar().to<float>();
               auto scaled_val = other * alpha;

               auto scaled_other_tensor = tensor_to_const(ctx, torch::tensor({scaled_val}));
               auto sub = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kSUB, self, scaled_other_tensor, util::node_info(n));
               TORCHTRT_CHECK(sub, "Unable to create sub layer from node: " << *n);
               sub->setName(util::node_info(n).c_str());
               LOG_DEBUG("Output tensor shape: " << sub->getOutput(0)->getDimensions());
               ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));

               return true;
             }})
        .pattern(
            {"aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar "
             "alpha=1) -> (Tensor(a!))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self - alpha * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto scalar = args[2].unwrapToScalar();

               if (1 != scalar.to<float>()) {
                 auto alphaTensor = scalar_to_tensor(ctx, scalar);
                 auto scaleLayer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     other,
                     alphaTensor,
                     util::node_info(n) + std::string("_AlphaMultiplier"));
                 TORCHTRT_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                 other = scaleLayer->getOutput(0);
               }

               auto sub = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, self, other, util::node_info(n));
               TORCHTRT_CHECK(sub, "Unable to create sub layer from node: " << *n);

               sub->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement other - alpha * self
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto scalar = args[2].unwrapToScalar();

               if (1 != scalar.to<float>()) {
                 auto alphaTensor = scalar_to_tensor(ctx, scalar);
                 auto scaleLayer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     self,
                     alphaTensor,
                     util::node_info(n) + std::string("_AlphaMultiplier"));
                 TORCHTRT_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                 self = scaleLayer->getOutput(0);
               }

               auto rsub = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, other, self, util::node_info(n));
               TORCHTRT_CHECK(rsub, "Unable to create rsub layer from node: " << *n);

               rsub->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], rsub->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::rsub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement other - alpha * self
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto scalar = args[2].unwrapToScalar();

               if (1 != scalar.to<float>()) {
                 auto alphaTensor = scalar_to_tensor(ctx, scalar);
                 auto scaleLayer = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     self,
                     alphaTensor,
                     util::node_info(n) + std::string("_AlphaMultiplier"));
                 TORCHTRT_CHECK(scaleLayer, "Unable to create alpha*input layer from node: " << *n);
                 self = scaleLayer->getOutput(0);
               }

               auto rsub = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, other, self, util::node_info(n));
               TORCHTRT_CHECK(rsub, "Unable to create rsub layer from node: " << *n);

               rsub->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], rsub->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::div.Tensor(Tensor self, Tensor other) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self / other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));

               TORCHTRT_CHECK(div, "Unable to create div layer from node: " << *n);

               div->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self / other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               std::string rounding_mode = "default";
               if (args[2].isIValue() && args[2].IValue()->isString()) {
                 rounding_mode = args[2].unwrapToString();
               }
               nvinfer1::ILayer* div = nullptr;
               if (rounding_mode == "floor") {
                 div =
                     add_elementwise(ctx, nvinfer1::ElementWiseOperation::kFLOOR_DIV, self, other, util::node_info(n));
               } else if (rounding_mode == "trunc") {
                 // trunc = floor(abs(div)) * sign(div)
                 auto tmp_div = add_elementwise(
                     ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n) + "_tmp_div");
                 auto abs = add_abs(ctx, n, tmp_div->getOutput(0), util::node_info(n) + "_absolute_val");

                 // In this case, we allow the floor unary on non-TRT Unary types, as it is needed for this
                 // specific function. Floor applied to non-float types equates to identity
                 nvinfer1::ITensor* floor;

                 if ((abs->getType() == nvinfer1::DataType::kINT32) || (abs->getType() == nvinfer1::DataType::kBOOL)) {
                   LOG_DEBUG(
                       "Tensor is of unsupported type " << abs->getType()
                                                        << " for IUnaryLayer::kFLOOR. Using identity instead.");
                   floor = abs;
                 } else {
                   auto floor_layer = ctx->net->addUnary(*abs, nvinfer1::UnaryOperation::kFLOOR);
                   TORCHTRT_CHECK(floor_layer, "Unable to create floor layer from node: " << *n);
                   floor_layer->setName((util::node_info(n) + "_floor").c_str());
                   floor = floor_layer->getOutput(0);
                 }

                 auto sign = ctx->net->addUnary(*tmp_div->getOutput(0), nvinfer1::UnaryOperation::kSIGN);
                 div = add_elementwise(
                     ctx, nvinfer1::ElementWiseOperation::kPROD, floor, sign->getOutput(0), util::node_info(n));
               } else {
                 div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));
               }

               TORCHTRT_CHECK(div, "Unable to create div layer from node: " << *n);

               div->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::div.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));
               TORCHTRT_CHECK(div, "Unable to create div layer from node: " << *n);

               div->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));

               TORCHTRT_CHECK(div, "Unable to create div layer from node: " << *n);

               div->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n));
               TORCHTRT_CHECK(div, "Unable to create div layer from node: " << *n);

               div->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], div->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::square(Tensor self) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto mul = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, self, util::node_info(n));
               TORCHTRT_CHECK(mul, "Unable to create mul layer from node: " << *n);

               mul->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], mul->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Should implement self * other
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto mul = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, other, util::node_info(n));
               TORCHTRT_CHECK(mul, "Unable to create mul layer from node: " << *n);

               mul->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], mul->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::mul.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto mul = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, other, util::node_info(n));
               TORCHTRT_CHECK(mul, "Unable to create mul layer from node: " << *n);

               mul->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], mul->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto mul = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, other, util::node_info(n));
               TORCHTRT_CHECK(mul, "Unable to create mul layer from node: " << *n);

               mul->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], mul->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::ne.Tensor(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto equal = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kEQUAL,
                   self,
                   other,
                   util::node_info(n) + std::string("is_equal"));
               TORCHTRT_CHECK(equal, "Unable to create elementwise equal layer from node: " << *n);
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
               TORCHTRT_CHECK(sub, "Unable to create ne (not equal) layer from node: " << *n);

               sub->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
               LOG_DEBUG("Not equal layer output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::ne.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto equal = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kEQUAL,
                   self,
                   other,
                   util::node_info(n) + std::string("is_equal"));
               TORCHTRT_CHECK(equal, "Unable to create elementwise equal layer from node: " << *n);
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
               TORCHTRT_CHECK(sub, "Unable to create ne (not equal) layer from node: " << *n);

               sub->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], sub->getOutput(0));
               LOG_DEBUG("Not equal layer output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto exponent = args[1].ITensorOrFreeze(ctx);
               auto pow =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPOW, self, exponent, util::node_info(n));
               TORCHTRT_CHECK(pow, "Unable to create Power layer from node: " << *n);

               pow->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], pow->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto exponent = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto pow =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPOW, self, exponent, util::node_info(n));
               TORCHTRT_CHECK(pow, "Unable to create Power layer from node: " << *n);

               pow->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], pow->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::floor_divide(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto floor_divide =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kFLOOR_DIV, self, other, util::node_info(n));
               TORCHTRT_CHECK(floor_divide, "Unable to create floor_divide layer from node: " << *n);

               floor_divide->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], floor_divide->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::floor_divide.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               auto floor_divide =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kFLOOR_DIV, self, other, util::node_info(n));
               TORCHTRT_CHECK(floor_divide, "Unable to create floor_divide layer from node: " << *n);

               floor_divide->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], floor_divide->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::max.other(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto max = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMAX, self, other, util::node_info(n));
               TORCHTRT_CHECK(max, "Unable to create max layer from node: " << *n);

               max->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], max->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::min.other(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // TODO: Remove with functionalization
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto min = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMIN, self, other, util::node_info(n));
               TORCHTRT_CHECK(min, "Unable to create min layer from node: " << *n);

               min->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], min->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::gt.Tensor(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto gt =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n));
               TORCHTRT_CHECK(gt, "Unable to create greater layer from node: " << *n);

               gt->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gt->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::gt.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               if (self->getType() != other->getType()) {
                 other = castITensor(ctx, other, self->getType());
               }
               auto gt =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n));
               TORCHTRT_CHECK(gt, "Unable to create greater layer from node: " << *n);

               gt->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gt->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::lt.Tensor(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto lt = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n));
               TORCHTRT_CHECK(lt, "Unable to create less layer from node: " << *n);

               lt->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], lt->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::lt.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               if (self->getType() != other->getType()) {
                 other = castITensor(ctx, other, self->getType());
               }
               auto lt = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n));
               TORCHTRT_CHECK(lt, "Unable to create less layer from node: " << *n);

               lt->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], lt->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::eq.Tensor(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);
               auto eq = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n));
               TORCHTRT_CHECK(eq, "Unable to create equal layer from node: " << *n);

               eq->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], eq->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::eq.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               if (self->getType() == nvinfer1::DataType::kBOOL) {
                 auto otherScalar = args[1].unwrapToScalar().to<float>();
                 if (otherScalar == 0 || otherScalar == 1) {
                   LOG_DEBUG("Since input tensor is type bool, casting input tensor and scalar to int32");
                   other = castITensor(ctx, other, nvinfer1::DataType::kINT32);
                   self = castITensor(ctx, self, nvinfer1::DataType::kINT32);
                 } else {
                   LOG_WARNING("Input Tensor has type bool, but scalar is not 0 or 1. Found: " << otherScalar);
                 }
               }
               if (self->getType() != other->getType()) {
                 other = castITensor(ctx, other, self->getType());
               }
               auto eq = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n));
               TORCHTRT_CHECK(eq, "Unable to create equal layer from node: " << *n);

               eq->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], eq->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::ge.Tensor(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);

               auto greater = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n) + "_greater");
               TORCHTRT_CHECK(greater, "Unable to create Greater layer from node: " << *n);

               auto equal = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
               TORCHTRT_CHECK(equal, "Unable to create Equal layer from node: " << *n);

               auto or_op = ctx->net->addElementWise(
                   *greater->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

               TORCHTRT_CHECK(or_op, "Unable to create Or layer from node: " << *n);
               or_op->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::ge.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               if (self->getType() != other->getType()) {
                 other = castITensor(ctx, other, self->getType());
               }

               auto greater = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kGREATER, self, other, util::node_info(n) + "_greater");
               TORCHTRT_CHECK(greater, "Unable to create Greater layer from node: " << *n);

               auto equal = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
               TORCHTRT_CHECK(equal, "Unable to create Equal layer from node: " << *n);

               auto or_op = ctx->net->addElementWise(
                   *greater->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

               TORCHTRT_CHECK(or_op, "Unable to create Or layer from node: " << *n);
               or_op->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::le.Tensor(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);

               auto less = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n) + "_less");
               TORCHTRT_CHECK(less, "Unable to create Less layer from node: " << *n);

               auto equal = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
               TORCHTRT_CHECK(equal, "Unable to create Equal layer from node: " << *n);

               auto or_op = ctx->net->addElementWise(
                   *less->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

               TORCHTRT_CHECK(or_op, "Unable to create Or layer from node: " << *n);
               or_op->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::le.Scalar(Tensor self, Scalar other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = scalar_to_tensor(ctx, args[1].unwrapToScalar());
               if (self->getType() != other->getType()) {
                 other = castITensor(ctx, other, self->getType());
               }

               auto less = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kLESS, self, other, util::node_info(n) + "_less");
               TORCHTRT_CHECK(less, "Unable to create Less layer from node: " << *n);

               auto equal = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kEQUAL, self, other, util::node_info(n) + "_equal");
               TORCHTRT_CHECK(equal, "Unable to create Equal layer from node: " << *n);

               auto or_op = ctx->net->addElementWise(
                   *less->getOutput(0), *equal->getOutput(0), nvinfer1::ElementWiseOperation::kOR);

               TORCHTRT_CHECK(or_op, "Unable to create Or layer from node: " << *n);
               or_op->setName(util::node_info(n).c_str());
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], or_op->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::logical_and(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // torch.logical_and autocasts inputs to bool
               auto input_as_bool = [&](int idx) {
                 auto x = args[idx].ITensorOrFreeze(ctx);
                 if (x->getType() != nvinfer1::DataType::kBOOL) {
                   x = castITensor(
                       ctx, x, nvinfer1::DataType::kBOOL, (util::node_info(n) + "_bool_" + str(idx)).c_str());
                 }
                 return x;
               };
               auto self = input_as_bool(0);
               auto other = input_as_bool(1);

               auto and_layer =
                   add_elementwise(ctx, nvinfer1::ElementWiseOperation::kAND, self, other, util::node_info(n) + "_and");
               TORCHTRT_CHECK(and_layer, "Unable to create and layer from node: " << *n);
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], and_layer->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::atan2(Tensor self, Tensor other) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // Element-wise divide input Tensors, apply atan unary, apply quadrant correction
               auto self = args[0].ITensorOrFreeze(ctx);
               auto other = args[1].ITensorOrFreeze(ctx);

               // atan(self / other)
               auto intermediate_div = add_elementwise(
                   ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, util::node_info(n) + "_intermediate_div");
               auto atan2_intermediate =
                   ctx->net->addUnary(*intermediate_div->getOutput(0), nvinfer1::UnaryOperation::kATAN);

               // Constant tensors used for quadrant correction
               auto ZERO = tensor_to_const(ctx, torch::tensor({0.}));
               auto ONE = tensor_to_const(ctx, torch::tensor({1.}));
               auto TWO = tensor_to_const(ctx, torch::tensor({2.}));
               // Using PI float for TRT compatibility, however double is preferred for PyTorch
               auto PI = tensor_to_const(ctx, torch::tensor({c10::pi<float>}));

               // Quadrant correction is only needed when (other < 0) (elementwise)
               // In this scenario, the correction is +/- pi, depending on the sign of self (elementwise)

               // Full atan2 Formula is given by:
               // atan2(self, other) = atan(self / other) - (other < 0) * (2 * (self < 0) - 1) * pi

               // Mask of (other < 0)
               auto other_mask = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kLESS,
                   other,
                   ZERO,
                   util::node_info(n) + "_less_than_zero_other_mask");

               // Mask of (self < 0)
               auto self_mask = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kLESS,
                   self,
                   ZERO,
                   util::node_info(n) + "_greater_than_zero_self_mask");

               // Apply 2 * x - 1 to translate mask from {0, 1} to {-1, 1}
               self_mask = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kPROD,
                   self_mask->getOutput(0),
                   TWO,
                   util::node_info(n) + "_greater_than_zero_times_two_self_mask");
               self_mask = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kSUB,
                   self_mask->getOutput(0),
                   ONE,
                   util::node_info(n) + "_greater_than_zero_normalized_self_mask");

               // Multiply mask by pi
               self_mask = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kPROD,
                   self_mask->getOutput(0),
                   PI,
                   util::node_info(n) + "_greater_than_zero_times_pi_self_mask");

               // Take product of masks to generate correction term
               auto correction_term = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kPROD,
                   other_mask->getOutput(0),
                   self_mask->getOutput(0),
                   util::node_info(n) + "_correction_term");

               // Add correction term to atan(self/other) to obtain atan2(self, other)
               auto atan2 = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kSUB,
                   atan2_intermediate->getOutput(0),
                   correction_term->getOutput(0),
                   util::node_info(n) + "_corrected_atan2");

               TORCHTRT_CHECK(atan2, "Unable to create atan2 layer from node: " << *n);

               atan2->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], atan2->getOutput(0));

               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
