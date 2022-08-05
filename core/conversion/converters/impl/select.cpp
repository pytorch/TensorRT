#include <ATen/ATen.h>
#include <vector>
#include "NvInfer.h"
#include "c10/util/intrusive_ptr.h"
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool add_split(ConversionCtx* ctx, const torch::jit::Node* n, args& args, bool split_list, bool unbind) {
  auto in = args[0].ITensor();
  auto numOutputs = 1, numRemainder = 0, axis = 0;
  std::vector<int64_t> sizes;

  if (unbind) {
    axis = args[1].unwrapToInt();
    auto maxDim = static_cast<int64_t>(in->getDimensions().nbDims);
    axis = axis < 0 ? axis + maxDim : axis;
    numOutputs = in->getDimensions().d[axis];
    sizes.insert(sizes.end(), numOutputs, 1);
  } else {
    axis = args[2].unwrapToInt();
    auto inDimSize = in->getDimensions().d[axis];
    if (split_list) {
      sizes = args[1].unwrapToIntList().vec();
      numOutputs = sizes.size();
    } else {
      auto split_size = args[1].unwrapToInt();
      numOutputs = inDimSize / split_size;
      numRemainder = inDimSize % split_size;
      for (int64_t i = 0; i < numOutputs; i++) {
        sizes.push_back(split_size);
      }
      if (numRemainder) {
        numOutputs += 1;
        sizes.push_back(numRemainder);
      }
    }
  }

  LOG_DEBUG("Number of split outputs: " << numOutputs);

  c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
  c10::TypePtr elementType = lt->getElementType();
  auto list = c10::impl::GenericList(elementType);
  list.reserve(numOutputs);

  int start_idx = 0;
  for (int64_t i = 0; i < numOutputs; i++) {
    at::Tensor indices = torch::arange(start_idx, start_idx + sizes[i], 1).to(torch::kI32);
    auto indicesTensor = tensor_to_const(ctx, indices);

    auto gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
    auto gather_out = gather_layer->getOutput(0);

    auto tensor_holder = TensorContainer();
    tensor_holder.hold_tensor(gather_out);
    auto ival = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
    list.emplace_back(ival);

    start_idx = start_idx + sizes[i];
  }

  auto split_output_ivalue = std::move(torch::jit::IValue(list));
  ctx->AssociateValueAndIValue(n->outputs()[0], split_output_ivalue);

  return true;
}

nvinfer1::ITensor* roll(
    ConversionCtx* ctx,
    nvinfer1::ITensor* in,
    int shift,
    int dim,
    const std::vector<int64_t>& in_shape) {
  auto in_dim = in_shape[dim];

  auto start = (in_dim - shift) % in_dim;
  // Behavior of % is different in C++ vs Python for negative numbers. This
  // corrects the difference.
  if (start < 0) {
    start = start + in_dim;
  }
  at::Tensor index0 = at::arange(start, in_dim, 1, torch::kInt32);
  at::Tensor index;
  if (start == 0) {
    index = index0;
  } else {
    at::Tensor index1 = at::arange(start, torch::kInt32);
    index = at::cat({index0, index1}, 0);
  }
  auto index_tensor = tensor_to_const(ctx, index);
  auto gather_layer = ctx->net->addGather(*in, *index_tensor, dim);
  auto out = gather_layer->getOutput(0);
  return out;
}

auto select_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::select.int(Tensor(a) self, int dim, int index) -> (Tensor(a))",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    auto maxDim = static_cast<int64_t>(in->getDimensions().nbDims);
                    auto dim = args[1].unwrapToInt();
                    // Handle negative axis by refering to nbDims of input Tensor
                    dim = dim < 0 ? dim + maxDim : dim;
                    auto ind = (int32_t)args[2].unwrapToInt();
                    // Along the specified dimension, handle negative index by subtracting along length of dimension.
                    ind = ind < 0 ? ind + in->getDimensions().d[dim] : ind;
                    LOG_DEBUG("Gather input dimensions: " << in->getDimensions());
                    LOG_DEBUG("Dimension to select: " << dim);
                    LOG_DEBUG("Index: " << ind);

                    // index to access needs to be an at::Tensor
                    at::Tensor indices = torch::tensor({ind}).to(torch::kI32);
                    auto const_out = tensor_to_const(ctx, indices);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, dim);
                    TORCHTRT_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto out = gather_layer->getOutput(0);

                    LOG_DEBUG("Gather tensor shape: " << out->getDimensions());

                    if (out->getDimensions().nbDims != 1) {
                      // IShuffleLayer removes redundant dimensions
                      auto shuffle_layer = ctx->net->addShuffle(*out);
                      TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                      shuffle_layer->setReshapeDimensions(util::squeezeDims(out->getDimensions(), dim));
                      shuffle_layer->setName(util::node_info(n).c_str());
                      out = shuffle_layer->getOutput(0);
                    }

                    out = ctx->AssociateValueAndTensor(n->outputs()[0], out);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto axis = args[1].unwrapToInt();
                    auto start = (int32_t)args[2].unwrapToInt();
                    auto length = (int32_t)args[3].unwrapToInt();

                    // index to access needs to be an at::Tensor
                    at::Tensor indices = torch::arange(start, start + length, 1).to(torch::kI32);
                    auto weights = Weights(ctx, indices);

                    // IConstantLayer to convert indices from Weights to ITensor
                    auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                    TORCHTRT_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TORCHTRT_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                    shuffle_layer->setReshapeDimensions(util::unpadDims(gather_out->getDimensions()));
                    shuffle_layer->setName(util::node_info(n).c_str());
                    auto shuffle_out = shuffle_layer->getOutput(0);

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_out);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern({"aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto axis = args[1].unwrapToInt();
                    torch::Tensor start = args[2].IValue()->toTensor().to(torch::kI32);
                    int32_t startIdx = start.item().to<int32_t>();
                    auto length = (int32_t)args[3].unwrapToInt();

                    // index to access needs to be an at::Tensor
                    at::Tensor indices = torch::arange(startIdx, startIdx + length, 1).to(torch::kI32);
                    auto weights = Weights(ctx, indices);

                    // IConstantLayer to convert indices from Weights to ITensor
                    auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
                    TORCHTRT_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TORCHTRT_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
                    shuffle_layer->setReshapeDimensions(util::unpadDims(gather_out->getDimensions()));
                    shuffle_layer->setName(util::node_info(n).c_str());
                    auto shuffle_out = shuffle_layer->getOutput(0);

                    auto out = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_out);

                    LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                    return true;
                  }})
        .pattern(
            {"aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto embeddingTensor = args[0].ITensorOrFreeze(ctx);
               auto indicesTensor = args[1].ITensorOrFreeze(ctx);
               // Set datatype for indices tensor to INT32
               auto identity = ctx->net->addIdentity(*indicesTensor);
               identity->setOutputType(0, nvinfer1::DataType::kINT32);
               indicesTensor = identity->getOutput(0);

               // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
               auto gather_layer = ctx->net->addGather(*embeddingTensor, *indicesTensor, 0);
               TORCHTRT_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
               auto gather_out = gather_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gather_out);

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }})
        .pattern({"aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensor();
                    auto shifts = args[1].unwrapToIntList().vec();
                    auto dims = args[2].unwrapToIntList().vec();

                    TORCHTRT_CHECK(dims.size() == shifts.size(), "dims.size() should be equal to shifts.size()");
                    if (ctx->input_is_dynamic) {
                      TORCHTRT_THROW_ERROR("aten::roll is currently not support in dynamic input shape compilation");
                    } else {
                      auto in_shape = util::toVec(in->getDimensions());
                      for (size_t i = 0; i < dims.size(); i++) {
                        auto dim = dims[i] < 0 ? (in_shape.size() + dims[i]) : dims[i];
                        TORCHTRT_CHECK(dim < in_shape.size(), "Dimension out of range");
                        in = roll(ctx, in, shifts[i], dim, in_shape);
                      }
                      auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);

                      LOG_DEBUG("Output tensor shape: " << out->getDimensions());

                      return true;
                    }
                  }})
        .pattern(
            {"aten::index.Tensor(Tensor self, Tensor?[] indices) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto ts = args[1].IValue()->toListRef();

               std::vector<nvinfer1::ITensor*> tensors;
               for (auto t : ts) {
                 if (t.isTensor()) {
                   auto torch_tensor = t.toTensor();
                   tensors.push_back(tensor_to_const(ctx, torch_tensor));
                 } else {
                   auto cont = t.toCustomClass<TensorContainer>();
                   tensors.push_back(cont->tensor());
                 }
               }

               // In TorchScript, aten::index.Tensor indexes the self tensor along its each dimension by several
               // indexes. In this version of Torch-TensorRT, it can only receive one index tensor which means it only
               // indexes the self tensor along dimension 0.
               TORCHTRT_CHECK(
                   tensors.size() == 1,
                   "In this version of Torch-TensorRT, aten::index.Tensor can only receive one index tensor which means it only indexes the self tensor along dimension 0.");
               auto indicesTensor = tensors[0];
               // Set datatype for indices tensor to INT32
               auto identity = ctx->net->addIdentity(*indicesTensor);
               identity->setOutputType(0, nvinfer1::DataType::kINT32);
               indicesTensor = identity->getOutput(0);

               // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices
               // from
               auto gather_layer = ctx->net->addGather(*in, *indicesTensor, 0);
               TORCHTRT_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
               auto gather_out = gather_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gather_out);

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               int axis = args[1].unwrapToInt();
               int maxDim = static_cast<int32_t>(in->getDimensions().d[axis]);
               bool dynamic_shape = ctx->input_is_dynamic;
               auto input_dim = in->getDimensions();
               // add Shape Tensor
               auto ishape_layer = ctx->net->addShape(*in);
               auto ishape_tensor = ishape_layer->getOutput(0); // input shape
               std::string node_name = n->outputs()[0]->debugName().c_str();

               int startIdx = 0;
               auto startIdxIVal = args[2].IValue();
               if (!startIdxIVal->isNone()) {
                 startIdx =
                     startIdxIVal->toInt() > std::numeric_limits<int32_t>::max() ? maxDim : startIdxIVal->toInt();
                 startIdx = maxDim == -1 ? startIdx : std::min(startIdx, maxDim);
               }
               // Handle case when given tensor index is negative
               if (maxDim > 0) { // only for static shape
                 startIdx = (startIdx < 0) ? (maxDim + startIdx) : startIdx;
               }

               // Bound the end index to input tensor dimensions at specified axis
               int endIdx = maxDim; // -1 for dynamic shape
               auto endIdxIVal = args[3].IValue();
               if (!endIdxIVal->isNone()) {
                 int truncate_value =
                     endIdxIVal->toInt() > std::numeric_limits<int32_t>::max() ? maxDim : endIdxIVal->toInt();
                 endIdx = maxDim == -1 ? truncate_value : std::min(truncate_value, maxDim);
               }
               if (maxDim > 0) {
                 endIdx = (endIdx < 0) ? (maxDim + endIdx) : endIdx;
               }
               int step = args[4].unwrapToInt();

               // update start, end, stride for static shape
               int nbdims = in->getDimensions().nbDims;
               nvinfer1::Dims start_, size_, stride_;
               start_.nbDims = nbdims;
               size_.nbDims = nbdims;
               stride_.nbDims = nbdims;
               for (int i = 0; i < nbdims; i++) {
                 if (i == axis) {
                   start_.d[i] = startIdx;
                   size_.d[i] = (endIdx - startIdx - 1) / step + 1;
                   stride_.d[i] = step;
                 } else {
                   start_.d[i] = 0;
                   size_.d[i] = input_dim.d[i]; // for static
                   stride_.d[i] = 1;
                 }
               }
               auto slice_layer = ctx->net->addSlice(*in, start_, size_, stride_);

               if (dynamic_shape) { // dynamic shape
                 LOG_DEBUG("Using dynamic version of slice");
                 // start tensor
                 at::Tensor start_tensor = torch::zeros({nbdims}).to(torch::kI32);
                 ;
                 start_tensor[axis] = startIdx;
                 auto start_itensor = tensor_to_const(ctx, start_tensor);

                 // step tensor
                 at::Tensor stride_tensor = torch::ones({nbdims}).to(torch::kI32);
                 stride_tensor[axis] = step;
                 auto stride_itensor = tensor_to_const(ctx, stride_tensor);

                 // end tensor
                 at::Tensor end_tensor = torch::zeros({nbdims}).to(torch::kI32);
                 for (int i = 0; i < nbdims; i++) {
                   if (i == axis) {
                     end_tensor[i] = endIdx == -1 ? -1 : endIdx - 1;
                   } else {
                     end_tensor[i] = input_dim.d[i] == -1 ? -1 : input_dim.d[i] - 1;
                   }
                 }
                 auto end_itensor = tensor_to_const(ctx, end_tensor);

                 // update start and end
                 nvinfer1::ITensor* out_start;
                 nvinfer1::ITensor* out_end;
                 auto start_end =
                     normalize_start_and_end(ctx, ishape_tensor, start_itensor, end_itensor, nbdims, node_name);
                 out_start = start_end[0];
                 out_end = start_end[1];

                 // calculate size
                 auto size_itensor = get_slice_size(ctx, out_start, out_end, stride_itensor, nbdims, node_name);

                 // update slice layer
                 slice_layer->setInput(1, *out_start); // start
                 slice_layer->setInput(2, *size_itensor); // size, must be set if input is dynamic
               }
               auto slice_out = slice_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_out);
               LOG_DEBUG("Slice layer output shape: " << out->getDimensions());

               return true;
             }})
        .pattern({"aten::split(Tensor self, int[] split_sizes, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, true, false);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::split.sizes(Tensor(a -> *) self, int[] split_size, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, true, false);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, false, false);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, true, false);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::unbind.int(Tensor(a -> *) self, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, false, true);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> (Tensor)",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto self = args[0].ITensorOrFreeze(ctx);
                    auto mask = args[1].ITensorOrFreeze(ctx);
                    mask = addPadding(ctx, n, mask, self->getDimensions().nbDims, false, true);
                    auto val = args[2].unwrapToScalar().to<float>();
                    auto val_t = tensor_to_const(ctx, torch::full(util::toVec(self->getDimensions()), val));

                    TORCHTRT_CHECK(
                        util::broadcastable(self->getDimensions(), mask->getDimensions(), /*multidirectional=*/false),
                        "Self and mask tensors are not broadcastable");

                    auto new_layer = ctx->net->addSelect(*mask, *val_t, *self);
                    TORCHTRT_CHECK(new_layer, "Unable to create layer for aten::masked_fill");

                    new_layer->setName(util::node_info(n).c_str());

                    auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], new_layer->getOutput(0));
                    LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
                    return true;
                  }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
