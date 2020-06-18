#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto mm_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns()
  .pattern({
    "aten::matmul(Tensor self, Tensor other) -> (Tensor)",
    [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
      nvinfer1::ITensor* self;
      if (args[0].isIValue()) {
        auto t = args[0].unwrapToTensor();
        auto t_weights = Weights(ctx, t);
        auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
        TRTORCH_CHECK(const_layer, "Unable to freeze tensor self for node: " << *n);
        const_layer->setName((util::node_info(n) + " [Freeze Tensor(self)]").c_str());
        self = const_layer->getOutput(0);
      } else {
        self = args[0].ITensor();
      }
      LOG_DEBUG("self tensor shape: " << self->getDimensions());

      nvinfer1::ITensor* other;
      if (args[1].isIValue()) {
        auto t = args[1].unwrapToTensor();
        auto t_weights = Weights(ctx, t);
        auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
        TRTORCH_CHECK(const_layer, "Unable to freeze tensor other for node: " << *n);
        const_layer->setName((util::node_info(n) + " [Freeze Tensor(other)]").c_str());
        other = const_layer->getOutput(0);
      } else {
        other = args[1].ITensor();
      }
      LOG_DEBUG("other tensor shape: " << other->getDimensions());

      auto mm_layer = ctx->net->addMatrixMultiply(*self, nvinfer1::MatrixOperation::kNONE, *other, nvinfer1::MatrixOperation::kNONE);
      TRTORCH_CHECK(mm_layer, "Unable to create matrix multiplication node: " << *n);
      mm_layer->setName(util::node_info(n).c_str());
      auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mm_layer->getOutput(0));

      LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
      return true;
    }
  });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch