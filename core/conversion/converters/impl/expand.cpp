#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

nvinfer1::ITensor* concat(int max_rank, int old_rank, ConversionCtx* ctx, nvinfer1::ITensor* tensor) {
  if (max_rank - old_rank > 0) {
    torch::Tensor thOne = torch::tensor(std::vector<int32_t>(max_rank - old_rank, 1), torch::kInt32);
    auto one_tensor = tensor_to_const(ctx, thOne);
    auto in_shape_tensor = ctx->net->addShape(*tensor)->getOutput(0);
    nvinfer1::ITensor* const args[2] = {one_tensor, in_shape_tensor};
    return ctx->net->addConcatenation(args, 2)->getOutput(0);
  } else { // max_rank - old_rank == 0
    return ctx->net->addShape(*tensor)->getOutput(0);
  }
}

bool add_expand(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* in, nvinfer1::Dims expandedDims) {
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
  slice_layer->setName(util::node_info(n).c_str());

  auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_layer->getOutput(0));

  LOG_DEBUG("Expand layer output tensor shape: " << out->getDimensions());

  return true;
}

bool add_expand_dynamic(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* in,
    nvinfer1::ITensor* expandedDimsTensor,
    nvinfer1::Dims expandedDims,
    bool is_expand_layer) {
  auto input_dims = in->getDimensions();
  auto input_rank = in->getDimensions().nbDims;
  auto output_rank = expandedDims.nbDims;
  TORCHTRT_CHECK(
      input_rank <= output_rank,
      "Number of dimensions of the desired expansion must be greater than or equal to the number of input dimensions");

  /* TODO: When the inputs are dynamic, some dimensions of the inputs are indeterminate before setBindingDimensions. For
     these indeterminate dimensions, we don't validate the expansion. Eg: For an input of [3, -1], we omit the
     validation of the second dimension. Need to explore a better way to validate the expansion.
  */
  // Validate the expansion. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]
  for (int64_t i = expandedDims.nbDims - 1; i >= 0; --i) {
    int64_t offset = expandedDims.nbDims - 1 - i;
    int64_t dim = input_dims.nbDims - 1 - offset;
    int64_t size = (dim >= 0) ? input_dims.d[dim] : 1;
    int64_t targetSize = expandedDims.d[i];
    // Passing -1 as the size for a dimension means not changing the size of that dimension in expand layer.
    if (targetSize != -1) {
      if (size != targetSize) {
        // if size == -1, we can't validate the expansion before setBindingDimensions.
        if (!(size == -1 || size == 1)) {
          TORCHTRT_THROW_ERROR(
              "The expanded size of tensor (" << targetSize << ")"
                                              << " must match the existing size (" << size << ")"
                                              << " at dimension " << i);
        }
      }
    } else {
      // In dynamic expand layer, for the new dimensions, the size cannot be set to -1. Eg: an input of [3, 1] can be
      // expanded to [3, -1, 4] but not [-1, 3, 4].
      if (is_expand_layer && dim < 0) {
        TORCHTRT_THROW_ERROR(
            "The expanded size of the tensor (" << targetSize << ") isn't allowed in a leading, non-existing dimension "
                                                << i);
      }
    }
  }

  size_t max_rank = std::max(input_rank, output_rank);

  // Dimensions are right alignment. Eg: an input of [3, 1] and max_rank = 4, the result of concat is [1, 1, 3, 1]
  auto new_input_shape_tensor = concat(max_rank, input_rank, ctx, in);
  auto new_output_shape_tensor = expandedDimsTensor;

  // Add a reshape layer to expand dims
  auto shuffle = ctx->net->addShuffle(*in);
  shuffle->setInput(1, *new_input_shape_tensor);

  // Start the slicing from beginning of tensor since this is an expand layer
  std::vector<int64_t> start_vec(max_rank, 0);
  nvinfer1::Dims starts_dim = util::toDims(c10::IntArrayRef(start_vec));
  at::Tensor thStart = torch::tensor(util::toVec(starts_dim), torch::kInt32);
  auto starts = tensor_to_const(ctx, thStart);

  // compute sizes = max(x,y).
  auto sizes =
      ctx->net->addElementWise(*new_input_shape_tensor, *new_output_shape_tensor, nvinfer1::ElementWiseOperation::kMAX)
          ->getOutput(0);
  nvinfer1::Dims sizes_dim{-1, {}};
  sizes_dim.nbDims = max_rank;

  // Compute (x > 1 ? 1 : 0) for x in newDims, assuming positive x, using only TensorRT operations.
  // min(1, sub(input_shape, 1))
  torch::Tensor thOne = torch::tensor({1}, torch::kInt32);
  auto one_tensor = tensor_to_const(ctx, thOne);
  auto x_sub_one = ctx->net->addElementWise(*new_input_shape_tensor, *one_tensor, nvinfer1::ElementWiseOperation::kSUB)
                       ->getOutput(0);
  auto strides = ctx->net->addElementWise(*one_tensor, *x_sub_one, nvinfer1::ElementWiseOperation::kMIN)->getOutput(0);
  nvinfer1::Dims strides_dim{-1, {}};
  strides_dim.nbDims = max_rank;

  // Slice layer does the expansion in TRT. Desired output size is specified by sizes input at index 2.
  auto slice = ctx->net->addSlice(*shuffle->getOutput(0), starts_dim, sizes_dim, strides_dim);
  slice->setInput(1, *starts);
  slice->setInput(2, *sizes);
  slice->setInput(3, *strides);

  auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], slice->getOutput(0));

  LOG_DEBUG("Expand layer output tensor shape: " << out_tensor->getDimensions());

  return true;
}

auto expand_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto input_dims = in->getDimensions();
               auto expanded_size = args[1].unwrapToIntList();
               auto expandedDims = util::toDims(expanded_size);
               LOG_DEBUG("(expand layer) Expand input from " << input_dims << " to " << expandedDims);
               if (ctx->input_is_dynamic) {
                 at::Tensor thExpanded_size = torch::tensor(expanded_size.vec(), torch::kInt32);
                 auto expandedDimsTensor = tensor_to_const(ctx, thExpanded_size);
                 return add_expand_dynamic(ctx, n, in, expandedDimsTensor, expandedDims, true);
               } else {
                 return add_expand(ctx, n, in, expandedDims);
               }
             }})
        .pattern(
            {"aten::expand_as(Tensor(a) self, Tensor other) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto input_dims = in->getDimensions();
               auto targetTensor = args[1].ITensorOrFreeze(ctx);
               auto targetDims = targetTensor->getDimensions();
               LOG_DEBUG("(expand_as layer) Expand input from " << input_dims << " to " << targetDims);
               if (ctx->input_is_dynamic) {
                 return add_expand_dynamic(
                     ctx, n, in, ctx->net->addShape(*targetTensor)->getOutput(0), targetDims, false);
               } else {
                 return add_expand(ctx, n, in, targetDims);
               }
             }})
        .pattern(
            {"aten::repeat(Tensor self, int[] repeats) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto input_dims = in->getDimensions();
               auto repeats = args[1].unwrapToIntList().vec();
               int repeats_rank = repeats.size();
               TORCHTRT_CHECK(
                   repeats_rank >= input_dims.nbDims,
                   "Number of repeat dimensions cannot be smaller than number of input dimensions");
               auto num_expand_dims = repeats_rank - input_dims.nbDims;

               if (ctx->input_is_dynamic) {
                 int input_rank = input_dims.nbDims;
                 int output_rank = repeats_rank;
                 auto new_input_shape_tensor = concat(output_rank, input_rank, ctx, in);

                 // Add a reshape layer to expand dims
                 auto shuffle = ctx->net->addShuffle(*in);
                 shuffle->setInput(1, *new_input_shape_tensor);
                 in = shuffle->getOutput(0);
               } else {
                 if (num_expand_dims > 0) {
                   nvinfer1::Dims reshape_dims;
                   reshape_dims.nbDims = repeats.size();
                   for (int i = 0; i < num_expand_dims; i++) {
                     reshape_dims.d[i] = 1;
                   }
                   for (int i = 0; i < input_dims.nbDims; i++) {
                     reshape_dims.d[num_expand_dims + i] = input_dims.d[i];
                   }
                   // Add a reshape layer to expand dims
                   auto reshape_layer = ctx->net->addShuffle(*in);
                   reshape_layer->setReshapeDimensions(reshape_dims);
                   in = reshape_layer->getOutput(0);
                   LOG_DEBUG("Input reshaped to : " << in->getDimensions() << " from " << input_dims);
                 }
                 LOG_DEBUG("Repeats: " << repeats);
               }

               // Concat across all repeat axes.
               // TODO: Implementation might not be performant. Explore other strategies to improve performance.
               for (int i = repeats.size() - 1; i >= 0; --i) {
                 std::vector<nvinfer1::ITensor*> tensors_vec;
                 for (int j = 0; j < repeats[i]; j++) {
                   tensors_vec.push_back(in);
                 }
                 auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
                 concat_layer->setAxis(i);
                 in = concat_layer->getOutput(0);
               }

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);

               LOG_DEBUG("Repeat layer output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::repeat_interleave.self_int(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto repeats = args[1].unwrapToScalar().to<int>();

               auto input_shape = self->getDimensions();

               int dim;
               if (args[2].IValue()->isNone()) {
                 dim = 0;

                 // Flatten self tensor
                 int size;
                 if (ctx->input_is_dynamic) {
                   // Set size to -1 if input is dynamic
                   size = -1;
                 } else {
                   size = 1;
                   for (int i = 0; i < input_shape.nbDims; i++) {
                     size *= input_shape.d[i];
                   }
                 }
                 auto flatten = ctx->net->addShuffle(*self);
                 TORCHTRT_CHECK(flatten, "Unable to create shuffle layer from node: " << *n);
                 flatten->setReshapeDimensions(util::toDims(std::vector<int64_t>({size})));
                 self = flatten->getOutput(0);
                 input_shape = self->getDimensions();
               } else {
                 dim = args[2].unwrapToScalar().to<int>();
               }

               if (ctx->input_is_dynamic) {
                 int dynamic_dims = 0;
                 for (int idx = 0; idx < input_shape.nbDims; idx++) {
                   if (input_shape.d[idx] == -1) {
                     dynamic_dims++;
                   }
                 }

                 if (dynamic_dims > 1) {
                   TORCHTRT_THROW_ERROR(
                       "Repeat_interleave is currently not supported when target shape contains more than one dynamic dimension");
                 }
               }

               // Insert singleton dimension after desired repeat dimension
               std::vector<int64_t> repeat_shape_vec;
               for (int j = 0; j < input_shape.nbDims; j++) {
                 repeat_shape_vec.push_back(input_shape.d[j]);
                 if (j == dim) {
                   repeat_shape_vec.push_back(1);
                 }
               }
               auto expand = ctx->net->addShuffle(*self);
               TORCHTRT_CHECK(expand, "Unable to create shuffle layer from node: " << *n);
               auto repeat_shape_dims = util::toDims(repeat_shape_vec);
               expand->setReshapeDimensions(repeat_shape_dims);

               // Expand on newly created singleton dimension
               repeat_shape_dims.d[dim + 1] = repeats;
               std::vector<int64_t> start_vec(repeat_shape_dims.nbDims, 0);
               auto start_dims = util::toDims(start_vec);

               std::vector<int64_t> strides_vec(repeat_shape_dims.nbDims, 1);
               strides_vec[dim + 1] = 0;
               auto strides_dims = util::toDims(strides_vec);

               auto slice = ctx->net->addSlice(*expand->getOutput(0), start_dims, repeat_shape_dims, strides_dims);

               if (ctx->input_is_dynamic) {
                 auto start_tensor = tensor_to_const(ctx, torch::tensor(start_vec, torch::kInt32));

                 auto expand_output_shape = ctx->net->addShape(*expand->getOutput(0))->getOutput(0);
                 std::vector<int64_t> repeat_const_vec(repeat_shape_dims.nbDims, 1);
                 repeat_const_vec[dim + 1] = repeats;
                 auto repeat_const = tensor_to_const(ctx, torch::tensor(repeat_const_vec, torch::kInt32));
                 auto repeat_shape_tensor =
                     ctx->net
                         ->addElementWise(*expand_output_shape, *repeat_const, nvinfer1::ElementWiseOperation::kPROD)
                         ->getOutput(0);

                 auto strides_tensor = tensor_to_const(ctx, torch::tensor(strides_vec, torch::kInt32));
                 slice->setInput(1, *start_tensor);
                 slice->setInput(2, *repeat_shape_tensor);
                 slice->setInput(3, *strides_tensor);
               }

               // Collapse repeated dimension back into desired dimension
               std::vector<int64_t> collapse_shape_vec;
               for (int k = 0; k < repeat_shape_dims.nbDims; k++) {
                 if (k == dim) {
                   int64_t collapse_dim = repeat_shape_dims.d[k] * repeat_shape_dims.d[++k];
                   // Set dim size to -1 if repeat is being done on dynamic dim
                   collapse_dim = std::max(collapse_dim, (int64_t)-1);
                   collapse_shape_vec.push_back(collapse_dim);
                 } else {
                   collapse_shape_vec.push_back(repeat_shape_dims.d[k]);
                 }
               }
               auto collapse = ctx->net->addShuffle(*slice->getOutput(0));
               TORCHTRT_CHECK(collapse, "Unable to create shuffle layer from node: " << *n);
               collapse->setReshapeDimensions(util::toDims(collapse_shape_vec));

               collapse->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], collapse->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
