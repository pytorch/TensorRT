#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool min_max_dim(ConversionCtx* ctx, const torch::jit::Node* n, args& args, nvinfer1::TopKOperation topKOperation) {
  auto self = args[0].ITensorOrFreeze(ctx);
  auto dim = args[1].unwrapToInt();
  auto keep_dims = args[2].unwrapToBool();
  auto selfDim = util::toVec(self->getDimensions());
  if (dim < 0) {
    dim = selfDim.size() + dim;
  }
  bool int_input = self->getType() == nvinfer1::DataType::kINT32;
  if (int_input) {
    LOG_DEBUG("topk layer does not support int32 inputs, adding cast to float");
    self = castITensor(ctx, self, nvinfer1::DataType::kFLOAT, util::node_info(n) + "_input");
  }
  uint32_t reduce_axes_mask = 1 << dim;
  auto topk_layer = ctx->net->addTopK(*self, topKOperation, 1, reduce_axes_mask);
  TORCHTRT_CHECK(topk_layer, "Unable to create topk layer from node: " << *n);
  auto topk_dims = util::toVec(topk_layer->getOutput(0)->getDimensions());

  nvinfer1::ITensor* out0 = nullptr;
  nvinfer1::ITensor* out1 = nullptr;
  if (!keep_dims) {
    TORCHTRT_CHECK(topk_dims[dim] == 1, "Unexpected size in squeeze dimension. Expected: 1 Actual: " << topk_dims[dim]);
    auto squeeze_layer = ctx->net->addShuffle(*topk_layer->getOutput(0));
    squeeze_layer->setReshapeDimensions(util::squeezeDims(topk_layer->getOutput(0)->getDimensions(), dim));
    TORCHTRT_CHECK(squeeze_layer, "Unable to create squeeze_layer layer from node: " << *n);
    out0 = ctx->AssociateValueAndTensor(n->outputs()[0], squeeze_layer->getOutput(0));

    auto squeeze_layer_indices = ctx->net->addShuffle(*topk_layer->getOutput(1));
    squeeze_layer_indices->setReshapeDimensions(util::squeezeDims(topk_layer->getOutput(1)->getDimensions(), dim));
    TORCHTRT_CHECK(squeeze_layer_indices, "Unable to create squeeze_layer_indices layer from node: " << *n);
    out1 = ctx->AssociateValueAndTensor(n->outputs()[1], squeeze_layer_indices->getOutput(0));
  } else {
    out0 = ctx->AssociateValueAndTensor(n->outputs()[0], topk_layer->getOutput(0));
    out1 = ctx->AssociateValueAndTensor(n->outputs()[1], topk_layer->getOutput(1));
  }
  if (int_input) {
    LOG_DEBUG("Adding cast of topK layer output back to int32");
    out0 = castITensor(ctx, out0, nvinfer1::DataType::kINT32, util::node_info(n) + "_output");
  }
  LOG_DEBUG("Output tensor(0) shape: " << out0->getDimensions());
  LOG_DEBUG("Output tensor(1) shape: " << out1->getDimensions());

  return true;
}

bool arg_min_max(ConversionCtx* ctx, const torch::jit::Node* n, args& args, nvinfer1::TopKOperation topKOperation) {
  auto self = args[0].ITensorOrFreeze(ctx);
  auto dim = args[1].unwrapToInt();
  auto keep_dims = args[2].unwrapToBool();
  auto selfDim = util::toVec(self->getDimensions());
  if (dim < 0) {
    dim = selfDim.size() + dim;
  }
  if (self->getType() == nvinfer1::DataType::kINT32) {
    LOG_DEBUG("topk layer does not support int32 inputs, adding cast to float");
    self = castITensor(ctx, self, nvinfer1::DataType::kFLOAT, util::node_info(n) + "_input");
  }
  uint32_t reduce_axes_mask = 1 << dim;
  auto topk_layer = ctx->net->addTopK(*self, topKOperation, 1, reduce_axes_mask);
  TORCHTRT_CHECK(topk_layer, "Unable to create topk layer from node: " << *n);
  auto topk_dims = util::toVec(topk_layer->getOutput(0)->getDimensions());

  nvinfer1::ITensor* out = nullptr;
  if (!keep_dims) {
    TORCHTRT_CHECK(topk_dims[dim] == 1, "Unexpected size in squeeze dimension. Expected: 1 Actual: " << topk_dims[dim]);
    auto squeeze_layer_indices = ctx->net->addShuffle(*topk_layer->getOutput(1));
    squeeze_layer_indices->setReshapeDimensions(util::squeezeDims(topk_layer->getOutput(1)->getDimensions(), dim));
    TORCHTRT_CHECK(squeeze_layer_indices, "Unable to create squeeze_layer_indices layer from node: " << *n);
    out = ctx->AssociateValueAndTensor(n->outputs()[0], squeeze_layer_indices->getOutput(0));
  } else {
    out = ctx->AssociateValueAndTensor(n->outputs()[0], topk_layer->getOutput(1));
  }

  LOG_DEBUG("Output tensor shape: " << out->getDimensions());

  return true;
}

auto max_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return min_max_dim(ctx, n, args, nvinfer1::TopKOperation::kMAX);
             }})
        .pattern(
            {"aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return min_max_dim(ctx, n, args, nvinfer1::TopKOperation::kMIN);
             }})
        .pattern(
            {"aten::argmax(Tensor self, int dim, bool keepdim=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return arg_min_max(ctx, n, args, nvinfer1::TopKOperation::kMAX);
             }})
        .pattern(
            {"aten::argmin(Tensor self, int dim, bool keepdim=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               return arg_min_max(ctx, n, args, nvinfer1::TopKOperation::kMIN);
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
