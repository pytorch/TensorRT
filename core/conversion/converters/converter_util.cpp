#include "core/conversion/converters/converter_util.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {

nvinfer1::ITensor* addPadding(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* tensor,
    int nDim,
    bool trailing,
    bool use_zeros) {
  const auto dims = tensor->getDimensions();

  if (dims.nbDims < nDim) {
    auto newDims = dims;
    for (int dim = dims.nbDims; dim < nDim; ++dim) {
      newDims = util::unsqueezeDims(newDims, trailing ? dim : 0, 1, use_zeros);
    }

    LOG_DEBUG("Original shape: " << dims << ", reshaping to: " << newDims);
    auto shuffle_layer = ctx->net->addShuffle(*tensor);
    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer");
    shuffle_layer->setReshapeDimensions(newDims);
    shuffle_layer->setZeroIsPlaceholder(use_zeros);
    shuffle_layer->setName((util::node_info(n) + " [Reshape to " + util::toStr(newDims) + ']').c_str());
    return shuffle_layer->getOutput(0);
  } else {
    return tensor;
  }
}

nvinfer1::ITensor* addUnpadding(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* tensor,
    int nDim,
    bool trailing,
    bool use_zeros) {
  const auto dims = tensor->getDimensions();
  if (dims.nbDims > nDim) {
    auto newDims = dims;
    for (int dim = dims.nbDims; dim > nDim; --dim) {
      newDims = util::squeezeDims(newDims, trailing ? dim - 1 : 0);
    }
    LOG_DEBUG("Original shape: " << dims << ", reshaping to: " << newDims);
    auto shuffle_layer = ctx->net->addShuffle(*tensor);
    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer");
    shuffle_layer->setReshapeDimensions(newDims);
    shuffle_layer->setZeroIsPlaceholder(use_zeros);
    shuffle_layer->setName((util::node_info(n) + " [Reshape to " + util::toStr(newDims) + "]").c_str());
    return shuffle_layer->getOutput(0);
  } else {
    return tensor;
  }
}

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

nvinfer1::ITensor* castITensor(ConversionCtx* ctx, nvinfer1::ITensor* tensor, nvinfer1::DataType dtype) {
  if (tensor->getType() != dtype) {
    std::ostringstream tensor_id;
    tensor_id << reinterpret_cast<int*>(tensor);

    auto id_layer = ctx->net->addIdentity(*tensor);
    TRTORCH_CHECK(id_layer, "Unable to create identity layer for ITensor: " << tensor_id.str());
    auto casted_tensor = id_layer->getOutput(0);
    casted_tensor->setType(dtype);

    LOG_DEBUG(ctx->logger, "Casting ITensor " << tensor_id.str() << " from " << tensor->getType() << " to " << dtype);

    std::stringstream ss;
    ss << "[Cast ITensor " << tensor_id.str() << " from " << tensor->getType() << " to " << dtype << "]";
    id_layer->setName(ss.str().c_str());
    return casted_tensor;
  } else {
    return tensor;
  }
}

nvinfer1::ITensor* tensor_to_const(ConversionCtx* ctx, at::Tensor t) {
  bool post_freeze_cast = false;
  nvinfer1::DataType post_freeze_cast_type = nvinfer1::DataType::kFLOAT;
  // Other "unsupported weights types" can be added to this check here
  if (t.scalar_type() == at::kBool) {
    post_freeze_cast = true;
    auto type = util::ScalarTypeToTRTDataType(t.scalar_type());
    post_freeze_cast_type = type;
    LOG_DEBUG("To cast layer back to " << post_freeze_cast_type << " from int after freezing");
    t = t.to(at::kFloat);
  }

  auto weights = Weights();
  if ((t.scalar_type() == at::kLong || t.scalar_type() == at::kDouble) &&
      !ctx->settings.truncate_long_and_double) {
    TRTORCH_THROW_ERROR(
        "Unable to freeze tensor of type Int64/Float64 into constant layer, try to compile model with truncate_long_and_double enabled");
  } else if (t.scalar_type() == at::kLong && ctx->settings.truncate_long_and_double) {
    weights = converters::Weights(ctx, t.toType(at::kInt));
    LOG_WARNING("Truncating weight (constant in the graph) from Int64 to Int32");
  } else if (t.scalar_type() == at::kDouble && ctx->settings.truncate_long_and_double) {
    weights = converters::Weights(ctx, t.toType(at::kFloat));
    LOG_WARNING("Truncating weight (constant in the graph) from Float64 to Float32");
  } else {
    weights = Weights(ctx, t);
  }

  auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
  TRTORCH_CHECK(const_layer, "Unable to freeze tensor");

  auto out = const_layer->getOutput(0);

  std::ostringstream tensor_id;
  tensor_id << reinterpret_cast<int*>(out);

  LOG_DEBUG(ctx->logger, "Freezing tensor " << tensor_id.str() << " as an IConstantLayer");
  const_layer->setName(("[Freeze Tensor " + tensor_id.str() + " ]").c_str());

  if (post_freeze_cast) {
    out = castITensor(ctx, out, post_freeze_cast_type);
  }

  return out;
}

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
