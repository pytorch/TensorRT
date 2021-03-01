#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto mm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::matmul(Tensor self, Tensor other) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto self = args[0].ITensorOrFreeze(ctx);
       auto in_shape = util::toVec(self->getDimensions());
       LOG_DEBUG("self tensor shape: " << self->getDimensions());

       auto other = args[1].ITensorOrFreeze(ctx);
       auto other_shape = util::toVec(other->getDimensions());
       LOG_DEBUG("other tensor shape: " << other->getDimensions());

       // add support when self dims != other dims
       if (in_shape.size() != other_shape.size() && in_shape.size() > 2) {
         auto shuffle = ctx->net->addShuffle(*self);
         std::vector<int64_t> new_shape;
         int val = 1;
         for (int i = 0; i < (int)in_shape.size() - 1; i++) {
           val = val * in_shape[i];
         }
         new_shape.push_back(val);
         new_shape.push_back(in_shape[in_shape.size() - 1]);
         shuffle->setReshapeDimensions(util::toDims(new_shape));
         auto mm_layer = ctx->net->addMatrixMultiply(
             *shuffle->getOutput(0), nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);
         mm_layer->setName(util::node_info(n).c_str());
         std::vector<int64_t> out_shape;
         for (int i = 0; i < (int)in_shape.size() - 1; i++) {
           out_shape.push_back(in_shape[i]);
         }
         out_shape.push_back(other_shape[other_shape.size() - 1]);
         auto out_shuffle = ctx->net->addShuffle(*mm_layer->getOutput(0));
         out_shuffle->setReshapeDimensions(util::toDims(out_shape));
         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out_shuffle->getOutput(0));
         LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       } else {
         auto mm_layer = ctx->net->addMatrixMultiply(
             *self, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);
         TRTORCH_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);
         mm_layer->setName(util::node_info(n).c_str());
         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));
         LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       }

       return true;
     }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch