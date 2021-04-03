#include <algorithm>
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
       int max_dim = std::max(in_shape.size(), other_shape.size());
       int min_dim = std::min(in_shape.size(), other_shape.size());
       // add support when self dims != other dims
       if (in_shape.size() != other_shape.size()) {
         auto org_tensor = in_shape.size() > other_shape.size() ? other : self;
         auto shuffle = ctx->net->addShuffle(*org_tensor);

         int diff_dim = max_dim - min_dim;
         auto old_shape = in_shape.size() > other_shape.size() ? other_shape : in_shape;
         std::vector<int64_t> new_shape;
         for (int i = 0; i < diff_dim; i++) {
           new_shape.push_back(1);
         }
         for (int i = 0; i < min_dim; i++) {
           new_shape.push_back(old_shape[i]);
         }
         shuffle->setReshapeDimensions(util::toDims(new_shape));
         auto reshaped_tensor = shuffle->getOutput(0);
         auto new_self = in_shape.size() > other_shape.size() ? self : reshaped_tensor;
         auto new_other = other_shape.size() > in_shape.size() ? other : reshaped_tensor;
         auto mm_layer = ctx->net->addMatrixMultiply(
             *new_self, nvinfer1::MatrixOperation::kNONE, *new_other, nvinfer1::MatrixOperation::kNONE);
         mm_layer->setName(util::node_info(n).c_str());

         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));
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