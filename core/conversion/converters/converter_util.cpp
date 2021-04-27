#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {

nvinfer1::ITensor* addPadding(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int nDim, bool trailing, bool use_zeros) {
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

nvinfer1::ITensor* addUnpadding(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int nDim, bool trailing, bool use_zeros) {
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
    shuffle_layer->setName((util::node_info(n) + " [Reshape to " + util::toStr(newDims)).c_str() + ']');
    return shuffle_layer->getOutput(0);
  } else {
    return tensor;
  }
}

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
