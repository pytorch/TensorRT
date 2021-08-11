#include <ATen/ATen.h>
#include <vector>
#include "NvInfer.h"
#include "c10/util/intrusive_ptr.h"
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool add_split(ConversionCtx* ctx, const torch::jit::Node* n, args& args, bool split_list) {
  auto in = args[0].ITensor();
  auto axis = args[2].unwrapToInt();
  auto inDimSize = in->getDimensions().d[axis];
  auto numOutputs = 1, numRemainder = 0;
  std::vector<int64_t> sizes;

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

auto select_registrations TRTORCH_UNUSED =
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
                    TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto out = gather_layer->getOutput(0);

                    LOG_DEBUG("Gather tensor shape: " << out->getDimensions());

                    if (out->getDimensions().nbDims != 1) {
                      // IShuffleLayer removes redundant dimensions
                      auto shuffle_layer = ctx->net->addShuffle(*out);
                      TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
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
                    TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
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
                    TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
                    auto const_out = const_layer->getOutput(0);

                    // IGatherLayer takes in input tensor, the indices, and the axis
                    // of input tensor to take indices from
                    auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
                    TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
                    auto gather_out = gather_layer->getOutput(0);

                    // IShuffleLayer removes redundant dimensions
                    auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                    TRTORCH_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
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
               TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
               auto gather_out = gather_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gather_out);

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto axis = args[1].unwrapToInt();
               auto maxDim = static_cast<int64_t>(in->getDimensions().d[axis]);
               auto startIdx = 0;
               auto startIdxIVal = args[2].IValue();
               if (!startIdxIVal->isNone()) {
                 startIdx = startIdxIVal->toInt();
               }
               // Handle case when given tensor index is negative
               auto start = (startIdx < 0) ? (maxDim + startIdx) : startIdx;
               // Bound the end index to input tensor dimensions at specified axis
               auto endIdx = maxDim;
               auto endIdxIVal = args[3].IValue();
               if (!endIdxIVal->isNone()) {
                 endIdx = std::min(endIdxIVal->toInt(), maxDim);
               }
               auto end = (endIdx < 0) ? (maxDim + endIdx) : endIdx;
               auto step = args[4].unwrapToInt();

               if (startIdxIVal->isNone() && endIdxIVal->isNone()) { // for dynamic shape
                 auto layer = ctx->net->addIdentity(*in);
                 layer->setOutputType(0, in->getType());
                 auto out = ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
                 LOG_DEBUG("Slice layer output shape: " << out->getDimensions());
                 return true;
               }

               LOG_DEBUG("Start idx: " << start);
               LOG_DEBUG("End idx: " << end);

               // indices to be accessed need to be an at::Tensor
               at::Tensor indices = torch::arange(start, end, step).to(torch::kI32);
               auto weights = Weights(ctx, indices);

               // IConstantLayer to convert indices from Weights to ITensor
               auto const_layer = ctx->net->addConstant(weights.shape, weights.data);
               TRTORCH_CHECK(const_layer, "Unable to create constant layer from node: " << *n);
               auto const_out = const_layer->getOutput(0);

               // IGatherLayer takes in input tensor, the indices, and the axis of input tensor to take indices from
               auto gather_layer = ctx->net->addGather(*in, *const_out, axis);
               TRTORCH_CHECK(gather_layer, "Unable to create gather layer from node: " << *n);
               auto gather_out = gather_layer->getOutput(0);

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], gather_out);

               LOG_DEBUG("Slice layer output shape: " << out->getDimensions());

               return true;
             }})
        .pattern({"aten::split(Tensor self, int[] split_sizes, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, true);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, false);
                    LOG_DEBUG("Converted split op into a list of IValues");
                    return true;
                  }})
        .pattern({"aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    add_split(ctx, n, args, true);
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

                    TRTORCH_CHECK(
                        util::broadcastable(self->getDimensions(), mask->getDimensions(), /*multidirectional=*/false),
                        "Self and mask tensors are not broadcastable");

                    auto new_layer = ctx->net->addSelect(*mask, *val_t, *self);
                    TRTORCH_CHECK(new_layer, "Unable to create layer for aten::masked_fill");

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
} // namespace trtorch
