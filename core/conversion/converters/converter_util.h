#pragma once

#include <map>
#include <string>

#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/conversion/converters/Weights.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
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

nvinfer1::ILayer* add_elementwise(
    ConversionCtx* ctx,
    nvinfer1::ElementWiseOperation op,
    nvinfer1::ITensor* self,
    nvinfer1::ITensor* other,
    const std::string& name);

// Apply an identity operation on a tensor. Used in the case where an input is an output to a network.
nvinfer1::ITensor* applyIdentityOp(ConversionCtx* ctx, nvinfer1::ITensor* tensor, const std::string& name);

// If an ITensor is of a type not dtype, add an Identity layer to cast it to dtype
nvinfer1::ITensor* castITensor(ConversionCtx* ctx, nvinfer1::ITensor* tensor, nvinfer1::DataType dtype);

// Freeze an at::Tensor in a IConstant layer
nvinfer1::ITensor* tensor_to_const(ConversionCtx* ctx, at::Tensor t, const std::string& name = std::string());

nvinfer1::ITensor* toITensor(ConversionCtx* ctx, const torch::jit::Node* n, at::Tensor* input);

nvinfer1::ITensor* clamp(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* x,
                        nvinfer1::ITensor* lower_bound, nvinfer1::ITensor* upper_bound);

nvinfer1::ITensor* bump_if_negtive(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* input_dim,
                                   nvinfer1::ITensor* indices);

void update_start_and_end(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* in_shape, 
                        nvinfer1::ITensor* in_start, nvinfer1::ITensor* in_end,
                        nvinfer1::ITensor** out_start, nvinfer1::ITensor** out_end);

bool is_dynamic_shape(nvinfer1::ITensor* tensor);

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
