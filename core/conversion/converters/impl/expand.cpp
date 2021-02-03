#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"
#include "plugins/checkshape_plugin.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

nvinfer1::ILayer* create_plugin(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* inShape,
    nvinfer1::ITensor* expandShape,
    int32_t in_rank,
    int32_t expand_rank,
    const char* name) {
  auto creator = new plugins::CheckShapePluginCreator();
  std::vector<nvinfer1::PluginField> fields;
  nvinfer1::PluginField input_rank("input_rank", &in_rank, nvinfer1::PluginFieldType::kINT32, 1);
  nvinfer1::PluginField output_rank("expand_rank", &expand_rank, nvinfer1::PluginFieldType::kINT32, 1);
  fields.push_back(input_rank);
  fields.push_back(output_rank);
  nvinfer1::PluginFieldCollection collection;
  collection.nbFields = fields.size();
  collection.fields = fields.data();
  auto plugin = creator->createPlugin(name, &collection);

  nvinfer1::ITensor* inputs[] = {inShape, expandShape};
  auto expandShape_layer = ctx->net->addPluginV2(inputs, 2, *plugin);
  TRTORCH_CHECK(expandShape_layer, "Unable to create interpolation plugin from node" << *n);

  expandShape_layer->setName("CheckShapePlugin");
  return expandShape_layer;
}

void addSliceInput(nvinfer1::Dims& dims, int idx, ConversionCtx* ctx, nvinfer1::ISliceLayer* slice) {
  int32_t rank = static_cast<int32_t>(dims.nbDims);
  int32_t* tmp = new int32_t[rank];
  for (int i = 0; i < rank; i++)
    tmp[i] = dims.d[i];
  const nvinfer1::Dims d{1, {rank}};
  const nvinfer1::Weights w{nvinfer1::DataType::kINT32, tmp, rank};
  auto t = ctx->net->addConstant(d, w)->getOutput(0);
  slice->setInput(idx, *t);
}

nvinfer1::ITensor* vec2Tensor(int32_t* dim, int rank, ConversionCtx* ctx) {
  const nvinfer1::Dims d{1, {static_cast<int32_t>(rank)}};
  const nvinfer1::Weights w{nvinfer1::DataType::kINT32, dim, rank};
  return ctx->net->addConstant(d, w)->getOutput(0);
}

nvinfer1::ITensor* concat(int max_rank, int old_rank, ConversionCtx* ctx, nvinfer1::ITensor* tensor) {
  if (max_rank - old_rank > 0) {
    int32_t* tmp = new int32_t[max_rank - old_rank];
    for (int i = 0; i < (max_rank - old_rank); i++)
      tmp[i] = 1;
    auto max_rank_tensor = vec2Tensor(tmp, max_rank - old_rank, ctx);
    auto in_shape_tensor = ctx->net->addShape(*tensor)->getOutput(0);
    nvinfer1::ITensor* const args[2] = {max_rank_tensor, in_shape_tensor};
    return ctx->net->addConcatenation(args, 2)->getOutput(0);
  } else { // max_rank - old_rank == 0
    return ctx->net->addShape(*tensor)->getOutput(0);
  }
}

bool add_expand(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* in, nvinfer1::Dims expandedDims) {
  auto input_dims = in->getDimensions();
  TRTORCH_CHECK(
      input_dims.nbDims <= expandedDims.nbDims,
      "Number of dimensions of the desired expansion must be greater than or equal to the number of input dimensions");

  // Validate the expansion. Eg: an input of [3, 1] can be expanded to [1, 3, 4] but not [3, 4, 1]
  for (int i = expandedDims.nbDims - 1; i >= 0; --i) {
    int64_t offset = expandedDims.nbDims - 1 - i;
    int64_t dim = input_dims.nbDims - 1 - offset;
    int64_t size = (dim >= 0) ? input_dims.d[dim] : 1;
    int64_t targetSize = expandedDims.d[i];
    if (targetSize != -1) {
      if (size != targetSize) {
        if (size != 1) {
          TRTORCH_THROW_ERROR(
              "The expanded size of tensor (" << targetSize << ")"
                                              << " must match the existing size (" << size << ")"
                                              << " at dimension " << i);
        }
      }
    } else {
      if (dim < 0) {
        TRTORCH_THROW_ERROR(
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

  // Start the slicing from beginning of tensor since this is an expand layer
  std::vector<int64_t> start_vec(expandedDims.nbDims, 0);
  auto start_offset = util::toDims(c10::IntArrayRef(start_vec));

  // Set the stride of non singleton dimension to 1
  std::vector<int64_t> strides_vec(expandedDims.nbDims, 0);
  for (int i = 0; i < expandedDims.nbDims; i++) {
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
    nvinfer1::ITensor* expandedDimsTensor) {
  auto input_shape_tensor = ctx->net->addShape(*in)->getOutput(0);
  auto input_rank = in->getDimensions().nbDims;
  auto output_rank = expandedDimsTensor->getDimensions().d[0];
  TRTORCH_CHECK(
      input_rank <= output_rank,
      "Number of dimensions of the desired expansion must be greater than or equal to the number of input dimensions");

  // add a plugin to check expandedDimsTensor whether match input_shape_tensor
  auto expandShape_layer =
      create_plugin(ctx, n, input_shape_tensor, expandedDimsTensor, input_rank, output_rank, "expandShape");
  auto _tensor = expandShape_layer->getOutput(0);

  size_t max_rank = std::max(input_rank, output_rank);

  // Dimensions are right alignment
  auto new_input_shape_tensor = concat(max_rank, input_rank, ctx, in);
  // LOG_DEBUG("Expand layer output tensor shape: " << new_output_shape_tensor->getDimensions());
  auto new_output_shape_tensor = expandedDimsTensor;

  // Add a reshape layer to expand dims
  auto shuffle = ctx->net->addShuffle(*in);
  shuffle->setInput(1, *new_input_shape_tensor);

  // Start the slicing from beginning of tensor since this is an expand layer
  std::vector<int64_t> start_vec(max_rank, 0);
  nvinfer1::Dims starts_dim = util::toDims(c10::IntArrayRef(start_vec));

  // compute sizes = max(x,y).
  auto sizes =
      ctx->net->addElementWise(*new_input_shape_tensor, *new_output_shape_tensor, nvinfer1::ElementWiseOperation::kMAX)
          ->getOutput(0);
  nvinfer1::Dims sizes_dim{-1, {}};
  sizes_dim.nbDims = max_rank;

  // Compute (x > 1 ? 1 : 0) for x in newDims, assuming positive x, using only TensorRT operations.
  // min(1, sub(input_shape, 1))
  int32_t* one_vector_tmp = new int32_t[1];
  one_vector_tmp[0] = 1;
  auto one_vector = vec2Tensor(one_vector_tmp, 1, ctx);
  auto x_sub_one = ctx->net->addElementWise(*new_input_shape_tensor, *one_vector, nvinfer1::ElementWiseOperation::kSUB)
                       ->getOutput(0);
  auto strides = ctx->net->addElementWise(*one_vector, *x_sub_one, nvinfer1::ElementWiseOperation::kMIN)->getOutput(0);
  nvinfer1::Dims strides_dim{-1, {}};
  strides_dim.nbDims = max_rank;

  // Slice layer does the expansion in TRT. Desired output size is specified by expandedDimsTensor
  auto slice = ctx->net->addSlice(*shuffle->getOutput(0), starts_dim, sizes_dim, strides_dim);
  addSliceInput(starts_dim, 1, ctx, slice);
  slice->setInput(2, *sizes);
  slice->setInput(3, *strides);

  auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], slice->getOutput(0));

  LOG_DEBUG("Expand layer output tensor shape: " << out_tensor->getDimensions());

  return true;
}

auto expand_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto input_dims = in->getDimensions();
               auto expanded_size = args[1].unwrapToIntList();
               auto expandedDims = util::toDims(expanded_size);
               LOG_DEBUG("(expand layer) Expand input from " << input_dims << " to " << expandedDims);
               if (ctx->input_is_dynamic) {
                 int expanded_size_rank = static_cast<int>(expanded_size.size());
                 int32_t* tmp = new int32_t[expanded_size_rank];
                 for (int i = 0; i < expanded_size_rank; i++)
                   tmp[i] = expanded_size[i];
                 auto expandedDimsTensor = vec2Tensor(tmp, expanded_size_rank, ctx);
                 return add_expand_dynamic(ctx, n, in, expandedDimsTensor);
               } else {
                 return add_expand(ctx, n, in, expandedDims);
               }
             }})
        .pattern(
            {"aten::expand_as(Tensor(a) self, Tensor other) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto input_dims = in->getDimensions();
               auto targetTensor = args[1].ITensor();
               auto targetDims = targetTensor->getDimensions();
               LOG_DEBUG("(expand_as layer) Expand input from " << input_dims << " to " << targetDims);
               if (ctx->input_is_dynamic) {
                 return add_expand_dynamic(ctx, n, in, ctx->net->addShape(*targetTensor)->getOutput(0));
               } else {
                 return add_expand(ctx, n, in, targetDims);
               }
             }})
        .pattern(
            {"aten::repeat(Tensor self, int[] repeats) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto input_dims = in->getDimensions();
               auto repeats = args[1].unwrapToIntList().vec();
               int repeats_rank = repeats.size();
               TRTORCH_CHECK(
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

               LOG_DEBUG("Repeat layer output tensor shape: " << in->getDimensions());
               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch