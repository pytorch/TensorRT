#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converter_util.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace util {

  nvinfer1::ILayer* padTensorDim(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int nDim)
    {
        const auto dims = tensor->getDimensions();
	
        if (dims.nbDims < nDim)
        {
	    auto newDims = dims;
	    for (int dim = dims.nbDims; dim < nDim; ++dim)
	      newDims = unsqueezeDims(newDims, dim);
            LOG_DEBUG("Original shape: " << dims << ", unsqueezing to: " << newDims);
            auto shuffle_layer = ctx->net->addShuffle(*tensor);
            TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer");
            shuffle_layer->setReshapeDimensions(newDims);
            shuffle_layer->setZeroIsPlaceholder(true);
            shuffle_layer->setName((util::node_info(n)+ " : Unsqueeze "+ tensor->getName()
                                    + " "  + toStr(dims) + "->" + toStr(newDims)).c_str());
            return shuffle_layer;
        } else
            return nullptr;
    }

  nvinfer1::ILayer* unpadTensorDim(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int nDim)
    {
        const auto dims = tensor->getDimensions();
        if (dims.nbDims > nDim)
        {
	    auto newDims = dims;
	    for (int dim = dims.nbDims; dim > nDim; --dim)
	      newDims = squeezeDims(newDims, dim-1);
            LOG_DEBUG("Original shape: " << dims << ", squeezing to: " << newDims);
            auto shuffle_layer = ctx->net->addShuffle(*tensor);
            TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer");
            shuffle_layer->setReshapeDimensions(newDims);
            shuffle_layer->setZeroIsPlaceholder(true);
            shuffle_layer->setName((util::node_info(n)+ " : Squeeze "+ tensor->getName()
                                    + " " + toStr(dims) + "->" + toStr(newDims)).c_str());
            return shuffle_layer;
        } else
            return nullptr;
    }

} // namespace util
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
