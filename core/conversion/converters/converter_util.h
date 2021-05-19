#pragma once

#include <map>
#include <string>

#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/conversion/converters/Weights.h"
#include "core/conversion/var/Var.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {

// If nDim < tensor size, adds shuffle layer to pad tensor with 1s (at the end if trailing) and returns
// (nDim-dimensional) shuffle layer's output. Otherwise, does nothing and passes tensor through. use _zeros controls
// whether we should be using 0 instead of -1 on the shape.
nvinfer1::ITensor* addPadding(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* tensor,
    int nDim,
    bool trailing = true,
    bool use_zeros = true);

// If nDim < tensor size, adds shuffle layer to un-pad tensor (at the end if trailing) and returns (nDim-dimensional)
// shuffle layer's output Otherwise, does nothing and passes tensor through. use _zeros controls whether we should be
// using 0 instead of -1 on the shape.
nvinfer1::ITensor* addUnpadding(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* tensor,
    int nDim,
    bool trailing = true,
    bool use_zeros = true);

// Insert a cast layer and cast the input to the output_dtype
bool register_cast_layer_orig(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* input,
    nvinfer1::DataType output_dtype);

// Insert a cast layer and cast the input to the output_dtype
bool register_cast_layer(
    ConversionCtx* ctx,
    const std::string layer_name,
    nvinfer1::ITensor* input,
    nvinfer1::DataType output_dtype);

nvinfer1::ILayer* add_elementwise(
    ConversionCtx* ctx,
    nvinfer1::ElementWiseOperation op,
    nvinfer1::ITensor* self,
    nvinfer1::ITensor* other,
    const std::string& name);


} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
