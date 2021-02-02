#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

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
    if (size != targetSize) {
      if (size != 1) {
        TRTORCH_THROW_ERROR(
            "The expanded size of tensor (" << targetSize << ")"
                                            << " must match the existing size (" << size << ")"
                                            << " at dimension " << i);
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

auto expand_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> (Tensor(a))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto input_dims = in->getDimensions();
                    auto expanded_size = args[1].unwrapToIntList();
                    auto expandedDims = util::toDims(expanded_size);
                    LOG_DEBUG("(expand layer) Expand input from " << input_dims << " to " << expandedDims);
                    return add_expand(ctx, n, in, expandedDims);
                  }})
        .pattern({"aten::expand_as(Tensor(a) self, Tensor other) -> (Tensor(a))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    // TODO: Currently expand supports static shapes. Need to explore if the same code can be extended
                    // to dynamic expansion.
                    auto in = args[0].ITensor();
                    auto input_dims = in->getDimensions();
                    auto targetTensor = args[1].ITensor();
                    auto targetDims = targetTensor->getDimensions();
                    LOG_DEBUG("(expand_as layer) Expand input from " << input_dims << " to " << targetDims);
                    return add_expand(ctx, n, in, targetDims);
                  }})
        .pattern({"aten::repeat(Tensor self, int[] repeats) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto input_dims = in->getDimensions();
                    auto repeats = args[1].unwrapToIntList().vec();
                    TRTORCH_CHECK(
                        repeats.size() >= input_dims.nbDims,
                        "Number of repeat dimensions cannot be smaller than number of input dimensions");
                    auto num_expand_dims = repeats.size() - input_dims.nbDims;
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
