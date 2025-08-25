#include "core/conversion/converters/converter_util.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {

nvinfer1::ITensor* addPadding(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* tensor,
    int nDim,
    bool trailing,
    bool use_zeros,
    const std::string& name) {
  const auto dims = tensor->getDimensions();

  if (dims.nbDims < nDim) {
    auto newDims = dims;
    for (int dim = dims.nbDims; dim < nDim; ++dim) {
      newDims = util::unsqueezeDims(newDims, trailing ? dim : 0, 1, use_zeros);
    }

    LOG_DEBUG("Original shape: " << dims << ", reshaping to: " << newDims);
    auto shuffle_layer = ctx->net->addShuffle(*tensor);
    TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer");
    shuffle_layer->setReshapeDimensions(newDims);
    shuffle_layer->setZeroIsPlaceholder(use_zeros);
    if (name.size()) {
      shuffle_layer->setName(name.c_str());
    } else {
      shuffle_layer->setName((util::node_info(n) + " [Reshape to " + util::toStr(newDims) + ']').c_str());
    }
    return shuffle_layer->getOutput(0);
  } else {
    return tensor;
  }
}

nvinfer1::ITensor* getShapeOutput(ConversionCtx* ctx, nvinfer1::ITensor* input_tensor, const std::string& name) {
  nvinfer1::ITensor* input_shape = ctx->net->addShape(*input_tensor)->getOutput(0);
  input_shape = castITensor(ctx, input_shape, nvinfer1::DataType::kINT32, name);
  return input_shape;
}

nvinfer1::ITensor* addUnpadding(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* tensor,
    int nDim,
    bool trailing,
    bool use_zeros,
    const std::string& name) {
  const auto dims = tensor->getDimensions();
  if (dims.nbDims > nDim) {
    auto newDims = dims;
    for (int dim = dims.nbDims; dim > nDim; --dim) {
      newDims = util::squeezeDims(newDims, trailing ? dim - 1 : 0);
    }
    LOG_DEBUG("Original shape: " << dims << ", reshaping to: " << newDims);
    auto shuffle_layer = ctx->net->addShuffle(*tensor);
    TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer");
    shuffle_layer->setReshapeDimensions(newDims);
    shuffle_layer->setZeroIsPlaceholder(use_zeros);
    if (name.size()) {
      shuffle_layer->setName(name.c_str());
    } else {
      shuffle_layer->setName((util::node_info(n) + " [Reshape to " + util::toStr(newDims) + ']').c_str());
    }
    return shuffle_layer->getOutput(0);
  } else {
    return tensor;
  }
}

nvinfer1::DataType promote_types(nvinfer1::DataType type_a, nvinfer1::DataType type_b) {
  auto torch_type_a = util::TRTDataTypeToScalarType(type_a);
  auto torch_type_b = util::TRTDataTypeToScalarType(type_b);
  auto promo_type = at::promote_types(torch_type_a, torch_type_b);
  auto trt_promo_type = util::ScalarTypeToTRTDataType(promo_type);
  return trt_promo_type;
}

nvinfer1::ILayer* add_elementwise(
    ConversionCtx* ctx,
    nvinfer1::ElementWiseOperation op,
    nvinfer1::ITensor* self,
    nvinfer1::ITensor* other,
    const std::string& name) {
  if (self->getType() == nvinfer1::DataType::kFLOAT && other->getType() == nvinfer1::DataType::kINT32) {
    LOG_DEBUG("Type mismatch, casting other to " << self->getType());
    other = castITensor(ctx, other, self->getType(), name);
  } else if (self->getType() == nvinfer1::DataType::kINT32 && other->getType() == nvinfer1::DataType::kFLOAT) {
    LOG_DEBUG("Type mismatch, casting self to " << other->getType());
    self = castITensor(ctx, self, other->getType(), name);
  }
  // ensure self to have larger number of dimension
  bool swapSelfOther = false;
  if (self->getDimensions().nbDims < other->getDimensions().nbDims) {
    std::swap(self, other);
    swapSelfOther = true;
  }

  if (self->getType() != other->getType()) {
    LOG_DEBUG(
        "Type mismatch for inputs in element-wise operation " << name << ": " << self->getType() << ", "
                                                              << other->getType());
    auto promo_type = promote_types(self->getType(), other->getType());
    if (self->getType() != promo_type) {
      LOG_DEBUG(
          "Element-wise op type promotion adding cast from " << self->getType() << " to " << promo_type << " for layer "
                                                             << name);
      self = castITensor(ctx, self, promo_type, name);
    }
    if (other->getType() != promo_type) {
      LOG_DEBUG(
          "Element-wise op type promotion adding cast from " << other->getType() << " to " << promo_type
                                                             << " for layer " << name);
      other = castITensor(ctx, other, promo_type, name);
    }
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
      nvinfer1::ITensor* selfShape = getShapeOutput(ctx, self, std::string(name + "_shape_cast").c_str());
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

nvinfer1::ITensor* add_abs(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* self,
    const std::string& name) {
  nvinfer1::ILayer* absolute_value_layer;

  // Check if TRT Unary ops support the input type
  bool unary_supported_input = (self->getType() == nvinfer1::DataType::kFLOAT) ||
      (self->getType() == nvinfer1::DataType::kHALF) || (self->getType() == nvinfer1::DataType::kINT8);
  if (unary_supported_input) {
    absolute_value_layer = ctx->net->addUnary(*self, nvinfer1::UnaryOperation::kABS);
    TORCHTRT_CHECK(absolute_value_layer, "Unable to create abs layer from node: " << *n);
    absolute_value_layer->setName(name.c_str());
  } else {
    LOG_GRAPH(
        "Tensor is of unsupported type "
        << self->getType() << " for IUnaryLayer::kABS. Using backup implementation via IElementWise (max(x, -x)");
    // For types not supported by kABS, use an elementwise implementation abs(x) = max(x, -1 * x)
    at::Tensor neg_one = torch::full({1}, -1).to(util::TRTDataTypeToScalarType(self->getType()));
    auto neg_one_const = tensor_to_const(ctx, neg_one);
    auto neg_layer = add_elementwise(
        ctx, nvinfer1::ElementWiseOperation::kPROD, self, neg_one_const, util::node_info(n) + std::string("_Negation"));
    TORCHTRT_CHECK(neg_layer, "Unable to create prod layer from node: " << *n);
    absolute_value_layer =
        add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMAX, self, neg_layer->getOutput(0), name);
    TORCHTRT_CHECK(absolute_value_layer, "Unable to create max layer from node: " << *n);
  }

  return absolute_value_layer->getOutput(0);
}

nvinfer1::ITensor* applyIdentityOp(ConversionCtx* ctx, nvinfer1::ITensor* tensor, const std::string& tensor_name) {
  auto id_layer = ctx->net->addIdentity(*tensor);
  auto id_out_tensor = id_layer->getOutput(0);
  id_out_tensor->setName(tensor_name.c_str());
  return id_out_tensor;
}

nvinfer1::ITensor* castITensor(
    ConversionCtx* ctx,
    nvinfer1::ITensor* tensor,
    nvinfer1::DataType dtype,
    const std::string& layer_name_prefix) {
  if (tensor->getType() != dtype) {
    std::ostringstream tensor_id;
    tensor_id << reinterpret_cast<int*>(tensor);

    auto id_layer = ctx->net->addCast(*tensor, dtype);
    TORCHTRT_CHECK(id_layer, "Unable to create cast layer for ITensor: " << tensor_id.str());

    auto casted_tensor = id_layer->getOutput(0);
    LOG_DEBUG(ctx->logger, "Casting ITensor " << tensor_id.str() << " from " << tensor->getType() << " to " << dtype);

    std::stringstream ss;
    if (layer_name_prefix.size()) {
      ss << layer_name_prefix << " ";
    }
    ss << "[Cast ITensor " << tensor_id.str() << " from " << tensor->getType() << " to " << dtype << "]";
    id_layer->setName(ss.str().c_str());
    return casted_tensor;
  } else {
    return tensor;
  }
}

nvinfer1::ITensor* tensor_to_const(ConversionCtx* ctx, at::Tensor t, const std::string& name) {
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
  if ((t.scalar_type() == at::kLong || t.scalar_type() == at::kDouble) && !ctx->settings.truncate_long_and_double) {
    TORCHTRT_THROW_ERROR(
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
  TORCHTRT_CHECK(const_layer, "Unable to freeze tensor");

  auto out = const_layer->getOutput(0);

  std::ostringstream tensor_id;
  tensor_id << reinterpret_cast<int*>(out);
  std::string tensor_name;

  if (!name.empty()) {
    tensor_name = name;
  } else {
    tensor_name = tensor_id.str();
  }
  LOG_DEBUG(ctx->logger, "Freezing tensor " << tensor_name << " as an IConstantLayer");
  const_layer->setName(("[Freeze Tensor " + tensor_name + " ]").c_str());

  if (post_freeze_cast) {
    out = castITensor(ctx, out, post_freeze_cast_type);
  }

  return out;
}

// clamp x to [lower_bound, upper_bound]
nvinfer1::ITensor* clamp(
    ConversionCtx* ctx,
    nvinfer1::ITensor* x,
    nvinfer1::ITensor* lower_bound,
    nvinfer1::ITensor* upper_bound,
    std::string const& name) {
  auto max_layer = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMAX, x, lower_bound, "max layer for " + name);
  TORCHTRT_CHECK(max_layer, "Unable to create max layer for clamp");
  LOG_DEBUG(ctx->logger, "Create " << max_layer->getName() << " for clamp");
  auto max_itensor = max_layer->getOutput(0);

  auto min_layer =
      add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMIN, max_itensor, upper_bound, "min layer for " + name);
  TORCHTRT_CHECK(min_layer, "Unable to create min layer for clamp");
  LOG_DEBUG(ctx->logger, "Create " << min_layer->getName() << " for clamp");
  auto min_itensor = min_layer->getOutput(0);
  return min_itensor;
}

// clamp x to [0, input_dim]
nvinfer1::ITensor* clamp_to_input_dim(
    ConversionCtx* ctx,
    nvinfer1::ITensor* x,
    nvinfer1::ITensor* input_dim,
    int nbdims,
    std::string const& name) {
  auto zero = torch::zeros({nbdims}).to(torch::kI32);
  auto zero_itensor = tensor_to_const(ctx, zero);
  auto one = torch::ones({nbdims}).to(torch::kI32);
  auto one_itensor = tensor_to_const(ctx, one);

  auto upper_bound_layer =
      add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, input_dim, one_itensor, "sub layer for " + name);
  TORCHTRT_CHECK(upper_bound_layer, "Unable to create sub layer for clamp to inputDim");
  LOG_DEBUG(ctx->logger, "Create " << upper_bound_layer->getName() << " for clamp to inputDim");
  auto upper_bound = upper_bound_layer->getOutput(0);

  auto max_layer = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMAX, x, zero_itensor, "max layer for " + name);
  TORCHTRT_CHECK(max_layer, "Unable to create max_layer for clamp to inputDim");
  LOG_DEBUG(ctx->logger, "Create " << max_layer->getName() << " for clamp to inputDim");
  auto max_itensor = max_layer->getOutput(0);

  auto min_layer =
      add_elementwise(ctx, nvinfer1::ElementWiseOperation::kMIN, max_itensor, upper_bound, "min layer for " + name);
  TORCHTRT_CHECK(min_layer, "Unable to create min_layer for clamp to inputDim");
  LOG_DEBUG(ctx->logger, "Create " << min_layer->getName() << " for clamp to inputDim");
  auto min_itensor = min_layer->getOutput(0);
  return min_itensor;
}

// return indices < 0 ? inputDims + indices : indices
nvinfer1::ITensor* normalize_indices(
    ConversionCtx* ctx,
    nvinfer1::ITensor* input_dim,
    nvinfer1::ITensor* indices,
    int nbdims,
    std::string const& name) {
  auto zero = torch::zeros({nbdims}).to(torch::kI32);
  auto neg = -torch::ones({nbdims}).to(torch::kI32);
  auto zero_itensor = tensor_to_const(ctx, zero);
  auto neg_itensor = tensor_to_const(ctx, neg);
  // find the indices that = -1
  auto signs = clamp(ctx, indices, neg_itensor, zero_itensor, "clamp layer for " + name);
  // get the inputDim value where indices == -1, else 0
  auto mul = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kPROD, signs, input_dim, "prod layer for " + name);
  TORCHTRT_CHECK(mul, "Unable to create mul layer in normalize_indices");
  LOG_DEBUG(ctx->logger, "Create " << mul->getName() << " for normalize_indices");
  auto mul_itensor = mul->getOutput(0);

  // add the inputDim value to indices where indices == -1
  auto sub = add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, indices, mul_itensor, "sub layer for " + name);
  TORCHTRT_CHECK(sub, "Unable to create sub layer in normalize_indices");
  LOG_DEBUG(ctx->logger, "Create " << sub->getName() << " for normalize_indices");
  auto sub_itensor = sub->getOutput(0);
  return sub_itensor;
}

std::vector<nvinfer1::ITensor*> normalize_start_and_end(
    ConversionCtx* ctx,
    nvinfer1::ITensor* in_shape,
    nvinfer1::ITensor* in_start,
    nvinfer1::ITensor* in_end,
    int nbdims,
    std::string const& name) {
  auto start = normalize_indices(ctx, in_shape, in_start, nbdims, "normalize start of " + name);
  auto out_start = clamp_to_input_dim(ctx, start, in_shape, nbdims, "clamp start to inputDim for " + name);
  auto end = normalize_indices(ctx, in_shape, in_end, nbdims, "normalize end of " + name);
  auto out_end = clamp_to_input_dim(ctx, end, in_shape, nbdims, "clamp end to inputDim for " + name);
  std::vector<nvinfer1::ITensor*> outputs;
  outputs.push_back(out_start);
  outputs.push_back(out_end);
  return outputs;
}

// size = (end - start) / stride + 1, where range is [start, end], end is included
nvinfer1::ITensor* get_slice_size(
    ConversionCtx* ctx,
    nvinfer1::ITensor* start,
    nvinfer1::ITensor* end,
    nvinfer1::ITensor* stride,
    int nbdims,
    std::string const& name) {
  at::Tensor one_tensor = torch::ones({nbdims}).to(torch::kI32);
  auto one_itensor = tensor_to_const(ctx, one_tensor);

  auto sub_layer =
      add_elementwise(ctx, nvinfer1::ElementWiseOperation::kSUB, end, start, "get_slice_size sub layer for " + name);
  TORCHTRT_CHECK(sub_layer, "Unable to create sub layer in calculate_output_size");
  LOG_DEBUG(ctx->logger, "Create " << sub_layer->getName() << " for calculate_output_size");
  auto sub_itensor = sub_layer->getOutput(0);

  auto div_layer = add_elementwise(
      ctx, nvinfer1::ElementWiseOperation::kDIV, sub_itensor, stride, "get_slice_size div layer for " + name);
  TORCHTRT_CHECK(div_layer, "Unable to create div layer in calculate_output_size");
  LOG_DEBUG(ctx->logger, "Create " << div_layer->getName() << " for calculate_output_size");
  auto div_itensor = div_layer->getOutput(0);

  auto add_layer = add_elementwise(
      ctx, nvinfer1::ElementWiseOperation::kSUM, div_itensor, one_itensor, "get_slice_size sum layer for " + name);
  TORCHTRT_CHECK(add_layer, "Unable to create add layer in calculate_output_size");
  LOG_DEBUG(ctx->logger, "Create " << add_layer->getName() << " for calculate_output_size");
  auto size_itensor = add_layer->getOutput(0);

  return size_itensor;
}

nvinfer1::ITensor* scalar_to_tensor(ConversionCtx* ctx, at::Scalar s) {
  nvinfer1::ITensor* out;
  if (s.isIntegral(false)) {
    auto s_int = s.to<int64_t>();
    auto s_t = torch::tensor({s_int}).to(at::kInt);
    out = tensor_to_const(ctx, s_t);
  } else if (s.isBoolean()) {
    auto s_bool = s.to<bool>();
    auto s_t = torch::tensor({s_bool}).to(at::kBool);
    out = tensor_to_const(ctx, s_t);
  } else if (s.isFloatingPoint()) {
    auto other_float = s.to<float>();
    auto s_t = torch::tensor({other_float});
    out = tensor_to_const(ctx, s_t);
  } else {
    out = nullptr;
    TORCHTRT_THROW_ERROR("Unsupported data type for scalar. Found: (" << s.type() << ")");
  }
  return out;
}

nvinfer1::ITensor* add_expand(ConversionCtx* ctx, nvinfer1::ITensor* in, nvinfer1::Dims expandedDims) {
  auto input_dims = in->getDimensions();
  TORCHTRT_CHECK(
      input_dims.nbDims <= expandedDims.nbDims,
      "Number of dimensions of the desired expansion must be greater than or equal to the number of input dimensions");

  // Validate the expansion. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]
  for (int64_t i = expandedDims.nbDims - 1; i >= 0; --i) {
    int64_t offset = expandedDims.nbDims - 1 - i;
    int64_t dim = input_dims.nbDims - 1 - offset;
    int64_t size = (dim >= 0) ? input_dims.d[dim] : 1;
    int64_t targetSize = expandedDims.d[i];
    // In expand layer passing -1 as the size for a dimension means not changing the size of that dimension.
    if (targetSize != -1) {
      if (size != targetSize) {
        if (size != 1) {
          TORCHTRT_THROW_ERROR(
              "The expanded size of tensor (" << targetSize << ")"
                                              << " must match the existing size (" << size << ")"
                                              << " at dimension " << i);
        }
      }
    } else {
      // For the new dimensions, the size cannot be set to -1. Eg: an input of [3, 1] can be expanded to [3, -1, 4] but
      // not [-1, 3, 4].
      if (dim < 0) {
        TORCHTRT_THROW_ERROR(
            "The expanded size of the tensor (" << targetSize << ") isn't allowed in a leading, non-existing dimension "
                                                << i);
      } else {
        // in(3, 1), expand(3, -1, 4) -> expand(3, 3, 4)
        expandedDims.d[i] = input_dims.d[dim];
      }
    }
  }

  auto num_expand_dims = expandedDims.nbDims - input_dims.nbDims;
  if (num_expand_dims > 0) {
    nvinfer1::Dims reshape_dims;
    reshape_dims.nbDims = expandedDims.nbDims;
    for (int64_t i = 0; i < num_expand_dims; i++) {
      reshape_dims.d[i] = 1;
    }
    for (int64_t i = 0; i < input_dims.nbDims; i++) {
      reshape_dims.d[num_expand_dims + i] = input_dims.d[i];
    }
    // Add a reshape layer to expand dims
    auto reshape_layer = ctx->net->addShuffle(*in);
    reshape_layer->setReshapeDimensions(reshape_dims);
    in = reshape_layer->getOutput(0);
    LOG_DEBUG("Input reshaped to : " << in->getDimensions() << " from " << input_dims);
  }

  // Start the slicing from beginning of tensor since this is an expand layer
  std::vector<int64_t> start_vec(expandedDims.nbDims, 0);
  auto start_offset = util::toDims(c10::IntArrayRef(start_vec));

  // Set the stride of non singleton dimension to 1
  std::vector<int64_t> strides_vec(expandedDims.nbDims, 0);
  for (int64_t i = 0; i < expandedDims.nbDims; i++) {
    strides_vec[i] = (in->getDimensions().d[i] != 1);
  }

  auto strides = util::toDims(c10::IntArrayRef(strides_vec));
  // Slice layer does the expansion in TRT. Desired output size is specified by expandedDims
  auto slice_layer = ctx->net->addSlice(*in, start_offset, expandedDims, strides);
  LOG_DEBUG(ctx->logger, "Expand Tensor: " << in->getName());

  return slice_layer->getOutput(0);
}

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
