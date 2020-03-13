#pragma once
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

nvinfer1::ILayer* add_elementwise(ConversionCtx* ctx, nvinfer1::ElementWiseOperation op,  nvinfer1::ITensor* self, nvinfer1::ITensor* other, float scalar=1) {
    auto self_dims = self->getDimensions();
    auto other_dims = other->getDimensions();

    TRTORCH_CHECK(util::volume(self_dims) == util::volume(other_dims), "Found inputs to elementwise operation do not have the same number of elements:\n  Found: self " << self_dims << " other " << other_dims);
    
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
        
        auto scale = Weights(ctx, scalar);
        auto scaled = ctx->net->addScaleNd(*other, nvinfer1::ScaleMode::kUNIFORM, {}, scale.data, {}, 0);
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

auto element_wise_registrations = RegisterNodeConversionPatterns()
     .pattern({
            "aten::add.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                // Should implement self + alpha * other 
                auto self = args[0].ITensor();
                auto other = args[1].ITensor();
                auto scalar = args[2].unwrapToScalar().to<float>();
                auto add = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, scalar);
                add->setName(util::node_info(n).c_str());
                auto out_value = n->outputs()[0];
                auto out_tensor = add->getOutput(0);
                out_tensor->setName(out_value->debugName().c_str());
                ctx->value_tensor_map[out_value] = out_tensor;
                LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
                return true;
            }
     }).pattern({
            "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                // Should implement self + alpha * other 
                auto self = args[0].ITensor();
                auto other = args[1].ITensor();
                auto scalar = args[2].unwrapToScalar().to<float>();
                auto add = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUM, self, other, scalar);
                add->setName(util::node_info(n).c_str());
                auto out_value = n->outputs()[0];
                auto out_tensor = add->getOutput(0);
                out_tensor->setName(out_value->debugName().c_str());
                ctx->value_tensor_map[out_value] = out_tensor;
                LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
                return true;
            }
     }).pattern({
            "aten::sub.Tensor(Tensor self, Tensor other, Scalar alpha=1) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                // Should implement self - alpha * other
                auto self = args[0].ITensor();
                auto other = args[1].ITensor();
                auto scalar = args[2].unwrapToScalar().to<float>();
                auto sub = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, self, other, scalar);
                sub->setName(util::node_info(n).c_str());
                auto out_value = n->outputs()[0];
                auto out_tensor = sub->getOutput(0);
                out_tensor->setName(out_value->debugName().c_str());
                ctx->value_tensor_map[out_value] = out_tensor;
                LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
                return true;
            }
      }).pattern({
             "aten::div(Tensor self, Tensor other) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                 // Should implement self / other 
                 auto self = args[0].ITensor();
                 auto other = args[1].ITensor();
                 auto div = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kDIV, self, other);
                 div->setName(util::node_info(n).c_str());
                 auto out_value = n->outputs()[0];
                 auto out_tensor = div->getOutput(0);
                 out_tensor->setName(out_value->debugName().c_str());
                 ctx->value_tensor_map[out_value] = out_tensor;
                 LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
                return true;
             }
      }).pattern({
             "aten::mul(Tensor self, Tensor other) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                 // Should implement self * other 
                 auto self = args[0].ITensor();
                 auto other = args[1].ITensor();
                 auto mul = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, self, other);
                 mul->setName(util::node_info(n).c_str());
                 auto out_value = n->outputs()[0];
                 auto out_tensor = mul->getOutput(0);
                 out_tensor->setName(out_value->debugName().c_str());
                 ctx->value_tensor_map[out_value] = out_tensor;
                 LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
                return true;
             }
         });
                 
            
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // trtorch 
