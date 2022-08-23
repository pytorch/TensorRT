#include <torch/torch.h>
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

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
               auto scalar = args[2].unwrapToScalar().to<float>();
               auto other = args[1].ITensorOrFreeze(ctx);

               if (1 != scalar) {
                 auto alphaTensor = tensor_to_const(ctx, torch::tensor({scalar}));
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
               auto scalar = args[2].unwrapToScalar().to<float>();
               auto other = args[1].ITensorOrFreeze(ctx);

               if (1 != scalar) {
                 auto alphaTensor = tensor_to_const(ctx, torch::tensor({scalar}));
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
                 auto tmp_div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other, "tmp_div");
                 auto abs = ctx->net->addUnary(*tmp_div->getOutput(0), nvinfer1::UnaryOperation::kABS);
                 auto floor = ctx->net->addUnary(*abs->getOutput(0), nvinfer1::UnaryOperation::kFLOOR);
                 auto sign = ctx->net->addUnary(*tmp_div->getOutput(0), nvinfer1::UnaryOperation::kSIGN);
                 div = add_elementwise(
                     ctx,
                     nvinfer1::ElementWiseOperation::kPROD,
                     floor->getOutput(0),
                     sign->getOutput(0),
                     util::node_info(n));
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
               auto otherScalar = args[1].unwrapToScalar().to<float>();
               auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
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
               auto otherScalar = args[1].unwrapToScalar().to<float>();
               auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
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
               auto scalar = args[1].unwrapToScalar();
               nvinfer1::ITensor* scalar_tensor;
               if (self->getType() == nvinfer1::DataType::kFLOAT || self->getType() == nvinfer1::DataType::kHALF) {
                 scalar_tensor = tensor_to_const(ctx, torch::tensor({scalar.to<float>()}));
               } else {
                 scalar_tensor = tensor_to_const(ctx, torch::tensor({scalar.to<int>()}));
               }
               auto equal = add_elementwise(
                   ctx,
                   nvinfer1::ElementWiseOperation::kEQUAL,
                   self,
                   scalar_tensor,
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
               auto exponentScalar = args[1].unwrapToScalar().to<float>();
               auto exponent = tensor_to_const(ctx, torch::tensor({exponentScalar}));
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
               auto otherScalar = args[1].unwrapToScalar().to<float>();
               auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
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
               auto otherScalar = args[1].unwrapToScalar().to<float>();
               auto other = tensor_to_const(ctx, torch::tensor({otherScalar}));
               if (self->getType() == nvinfer1::DataType::kBOOL) {
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
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
