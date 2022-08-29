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
        .pattern(
            {"aten::select.int(Tensor(a) self, int dim, int index) -> (Tensor(a))",
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
                 shuffle_layer->setReshapeDimensions(
                     util::squeezeDims(out->getDimensions(), dim, !ctx->input_is_dynamic));
                 shuffle_layer->setName(util::node_info(n).c_str());
                 out = shuffle_layer->getOutput(0);
               }

               out = ctx->AssociateValueAndTensor(n->outputs()[0], out);

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)",
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
        .pattern(
            {"aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, int length) -> Tensor(a)",
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
        .pattern(
            {"aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto shifts = args[1].unwrapToIntList().vec();
               auto dims = args[2].unwrapToIntList().vec();

               TORCHTRT_CHECK(dims.size() == shifts.size(), "dims.size() should be equal to shifts.size()");
               auto in_shape = util::toVec(in->getDimensions());
               for (size_t i = 0; i < dims.size(); i++) {
                 auto dim = dims[i] < 0 ? (in_shape.size() + dims[i]) : dims[i];
                 TORCHTRT_CHECK(dim < in_shape.size(), "Dimension out of range");
                 TORCHTRT_CHECK(
                     in_shape[dim] != -1, "aten::roll is not supported when the targeted dimension is dynamic");
                 in = roll(ctx, in, shifts[i], dim, in_shape);
               }
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);

               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::index.Tensor(Tensor self, Tensor?[] indices) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               // refer to https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py#L4627
               auto in = args[0].ITensorOrFreeze(ctx);
               auto ts = args[1].IValue()->toListRef();

               std::vector<nvinfer1::ITensor*> tensors;
               std::vector<int32_t> adv_idx_indices;
               for (auto i = 0; i < ts.size(); i++) {
                 auto t = ts[i];
                 if (t.isTensor()) {
                   auto torch_tensor = t.toTensor().to(torch::kInt32);
                   tensors.push_back(tensor_to_const(ctx, torch_tensor));
                   adv_idx_indices.push_back(i);
                 } else {
                   // IValue
                   if (!t.isNone()) {
                     adv_idx_indices.push_back(i);
                     auto cont = t.toCustomClass<TensorContainer>();
                     // Set datatype for indices tensor to INT32
                     auto identity = ctx->net->addIdentity(*cont->tensor());
                     identity->setOutputType(0, nvinfer1::DataType::kINT32);
                     tensors.push_back(identity->getOutput(0));
                   }
                 }
               }

               if (tensors.size() == 0) {
                 auto identity_out = ctx->net->addIdentity(*in)->getOutput(0);
                 auto out = ctx->AssociateValueAndTensor(n->outputs()[0], identity_out);
                 LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               } else if (tensors.size() == 1) {
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
               } else {
                 auto inDims = in->getDimensions();
                 int rank = inDims.nbDims;
                 LOG_WARNING("If indices include negative values, the exported graph will produce incorrect results.");
                 int adv_idx_count = adv_idx_indices.size();
                 auto in_shape_itensor = ctx->net->addShape(*in)->getOutput(0);

                 std::vector<nvinfer1::ITensor*> dim_tensor_list;
                 for (int i = 0; i < rank; i++) {
                   auto dim_tensor =
                       ctx->net
                           ->addGather(*in_shape_itensor, *tensor_to_const(ctx, torch::tensor({i}, torch::kInt32)), 0)
                           ->getOutput(0);
                   dim_tensor_list.push_back(dim_tensor);
                 }

                 // t: [x_1, y_1, y_2, ..., x_m, ..., y_n] -> t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n],
                 // where t is a tensor of rank m+n, {x_i} are axes where tensor index is provided, and {y_i} are axes
                 // for ":".
                 auto in_transpose_layer = ctx->net->addShuffle(*in);
                 TORCHTRT_CHECK(in_transpose_layer, "Unable to create shuffle layer from node: " << *n);
                 nvinfer1::Permutation permute;
                 std::vector<int32_t> new_order;
                 for (int i = 0; i < adv_idx_count; i++) {
                   new_order.push_back(adv_idx_indices[i]);
                 }
                 for (int i = 0; i < rank; i++) {
                   if (std::find(adv_idx_indices.begin(), adv_idx_indices.end(), i) == adv_idx_indices.end()) {
                     new_order.push_back(i);
                   }
                 }
                 std::copy(new_order.begin(), new_order.end(), permute.order);
                 in_transpose_layer->setSecondTranspose(permute);
                 auto shuffle_out = in_transpose_layer->getOutput(0);

                 //  t: [x_1, x_2, ..., x_m, y_1, y_2, ..., y_n] -> t: [x_1*x_2* ...*x_m, y_1*y_2* ...*y_n]
                 nvinfer1::ITensor* flatten_tensor = NULL;
                 {
                   auto shuffle_shape_tensor = ctx->net->addShape(*shuffle_out)->getOutput(0);
                   auto d0 = tensor_to_const(ctx, torch::tensor({1}, torch::kInt32));
                   for (int i = 0; i < adv_idx_count; i++) {
                     auto dim_tensor =
                         ctx->net
                             ->addGather(
                                 *shuffle_shape_tensor, *tensor_to_const(ctx, torch::tensor({i}, torch::kInt32)), 0)
                             ->getOutput(0);
                     d0 = add_elementwise(
                              ctx,
                              nvinfer1::ElementWiseOperation::kPROD,
                              d0,
                              dim_tensor,
                              std::string("compute_dim0_") + std::to_string(i))
                              ->getOutput(0);
                   }

                   auto d1 = tensor_to_const(ctx, torch::tensor({1}, torch::kInt32));
                   for (int i = adv_idx_count; i < rank; i++) {
                     auto dim_tensor =
                         ctx->net
                             ->addGather(
                                 *shuffle_shape_tensor, *tensor_to_const(ctx, torch::tensor({i}, torch::kInt32)), 0)
                             ->getOutput(0);
                     d1 = add_elementwise(
                              ctx,
                              nvinfer1::ElementWiseOperation::kPROD,
                              d1,
                              dim_tensor,
                              std::string("compute_dim1_") + std::to_string(i))
                              ->getOutput(0);
                   }

                   std::vector<nvinfer1::ITensor*> concat_tensors;
                   concat_tensors.push_back(d0);
                   concat_tensors.push_back(d1);
                   auto concat_layer = ctx->net->addConcatenation(concat_tensors.data(), concat_tensors.size());

                   auto shuffle = ctx->net->addShuffle(*shuffle_out);
                   shuffle->setInput(1, *concat_layer->getOutput(0));
                   flatten_tensor = shuffle->getOutput(0);
                   LOG_DEBUG(flatten_tensor->getDimensions());
                 }

                 // tensor index = \sum_{i=1}^m (ind_i * \prod_{j=i+1}^m (x_j)),  ind_i is input indices[i], x_j is the
                 // j dimension of input x.
                 nvinfer1::ITensor* multiplier = dim_tensor_list[adv_idx_indices[adv_idx_count - 1]];
                 nvinfer1::ITensor* cum_adv_index = tensors[adv_idx_count - 1];
                 for (int i = adv_idx_count - 2; i >= 0; i--) {
                   nvinfer1::ITensor* adv_index = add_elementwise(
                                                      ctx,
                                                      nvinfer1::ElementWiseOperation::kPROD,
                                                      tensors[i],
                                                      multiplier,
                                                      std::string("adv_index_") + std::to_string(i))
                                                      ->getOutput(0);
                   cum_adv_index = add_elementwise(
                                       ctx,
                                       nvinfer1::ElementWiseOperation::kSUM,
                                       cum_adv_index,
                                       adv_index,
                                       std::string("cum_adv_index_") + std::to_string(i))
                                       ->getOutput(0);
                   multiplier = add_elementwise(
                                    ctx,
                                    nvinfer1::ElementWiseOperation::kPROD,
                                    multiplier,
                                    dim_tensor_list[adv_idx_indices[i]],
                                    std::string("multiplier_") + std::to_string(i))
                                    ->getOutput(0);
                 }

                 // perform gather
                 auto gather_out = ctx->net->addGather(*flatten_tensor, *cum_adv_index, 0)->getOutput(0);

                 nvinfer1::ITensor* reshape_output = NULL;
                 {
                   auto cum_adv_index_shape_tensor = ctx->net->addShape(*cum_adv_index)->getOutput(0);
                   // check if all advanced indices are consecutive.
                   if (adv_idx_count == (adv_idx_indices[adv_idx_count - 1] - adv_idx_indices[0] + 1)) {
                     // unfold regular index axes
                     std::vector<nvinfer1::ITensor*> concat_tensors;
                     concat_tensors.push_back(tensor_to_const(ctx, torch::tensor({-1}, torch::kInt32)));
                     for (int i = 0; i < rank; i++) {
                       if (std::find(adv_idx_indices.begin(), adv_idx_indices.end(), i) == adv_idx_indices.end()) {
                         nvinfer1::ITensor* current_dim = dim_tensor_list[i];
                         concat_tensors.push_back(current_dim);
                       }
                     }
                     auto concat_layer = ctx->net->addConcatenation(concat_tensors.data(), concat_tensors.size());
                     auto regular_index_shuffle_layer = ctx->net->addShuffle(*gather_out);
                     regular_index_shuffle_layer->setInput(1, *concat_layer->getOutput(0));
                     auto unfold_tensor = regular_index_shuffle_layer->getOutput(0);

                     // Transpose folded advanced indexed axis to its original location.
                     auto transpose_advanced_shuffle_layer = ctx->net->addShuffle(*unfold_tensor);
                     nvinfer1::Permutation permute;
                     std::vector<int32_t> new_order;
                     for (int i = 1; i < adv_idx_indices[0] + 1; i++) {
                       new_order.push_back(i);
                     }
                     new_order.push_back(0);
                     for (int i = adv_idx_indices[0] + 1; i < rank - adv_idx_count + 1; i++) {
                       new_order.push_back(i);
                     }
                     std::copy(new_order.begin(), new_order.end(), permute.order);
                     transpose_advanced_shuffle_layer->setSecondTranspose(permute);
                     auto shuffle_out = transpose_advanced_shuffle_layer->getOutput(0);

                     // unfold advanced index axes
                     std::vector<nvinfer1::ITensor*> concat_final_tensors;
                     for (int i = 0; i < adv_idx_indices[0]; i++) {
                       nvinfer1::ITensor* current_dim = dim_tensor_list[i];
                       concat_final_tensors.push_back(current_dim);
                     }
                     concat_final_tensors.push_back(cum_adv_index_shape_tensor);
                     for (int i = adv_idx_indices[0]; i < rank; i++) {
                       if (std::find(adv_idx_indices.begin(), adv_idx_indices.end(), i) == adv_idx_indices.end()) {
                         nvinfer1::ITensor* current_dim = dim_tensor_list[i];
                         concat_final_tensors.push_back(current_dim);
                       }
                     }
                     auto concat_final_shape_layer =
                         ctx->net->addConcatenation(concat_final_tensors.data(), concat_final_tensors.size());
                     auto unfold_advanced_shuffle_layer = ctx->net->addShuffle(*shuffle_out);
                     unfold_advanced_shuffle_layer->setInput(1, *concat_final_shape_layer->getOutput(0));
                     reshape_output = unfold_advanced_shuffle_layer->getOutput(0);
                   } else {
                     std::vector<nvinfer1::ITensor*> concat_tensors;
                     concat_tensors.push_back(cum_adv_index_shape_tensor);
                     for (int i = 0; i < rank; i++) {
                       if (std::find(adv_idx_indices.begin(), adv_idx_indices.end(), i) == adv_idx_indices.end()) {
                         nvinfer1::ITensor* current_dim = dim_tensor_list[i];
                         concat_tensors.push_back(current_dim);
                       }
                     }
                     auto concat_layer = ctx->net->addConcatenation(concat_tensors.data(), concat_tensors.size());
                     auto shuffle_layer = ctx->net->addShuffle(*gather_out);
                     shuffle_layer->setInput(1, *concat_layer->getOutput(0));
                     reshape_output = shuffle_layer->getOutput(0);
                   }
                 }

                 auto out = ctx->AssociateValueAndTensor(n->outputs()[0], reshape_output);
                 LOG_DEBUG("Output tensor shape: " << out->getDimensions());
               }
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
               if (!dynamic_shape) {
                 auto slice_layer = ctx->net->addSlice(*in, start_, size_, stride_);
                 LOG_DEBUG("start_:" << start_);
                 LOG_DEBUG("size_:" << size_);
                 LOG_DEBUG("stride_:" << stride_);
                 auto slice_out = slice_layer->getOutput(0);
                 auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_out);
                 LOG_DEBUG("Slice layer output shape: " << out->getDimensions());
               } else { // dynamic shape
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
                 auto slice_layer = ctx->net->addSlice(*in, start_, size_, stride_);
                 slice_layer->setInput(1, *out_start); // start
                 slice_layer->setInput(2, *size_itensor); // size, must be set if input is dynamic
                 auto slice_out = slice_layer->getOutput(0);
                 auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_out);
                 LOG_DEBUG("Slice layer output shape: " << out->getDimensions());
               }

               return true;
             }})
        .pattern(
            {"aten::split(Tensor self, int[] split_sizes, int dim=0) -> (Tensor[])",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               add_split(ctx, n, args, true, false);
               LOG_DEBUG("Converted split op into a list of IValues");
               return true;
             }})
        .pattern(
            {"aten::split.sizes(Tensor(a -> *) self, int[] split_size, int dim=0) -> (Tensor[])",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               add_split(ctx, n, args, true, false);
               LOG_DEBUG("Converted split op into a list of IValues");
               return true;
             }})
        .pattern(
            {"aten::split.Tensor(Tensor(a) self, int split_size, int dim=0) -> (Tensor[])",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               add_split(ctx, n, args, false, false);
               LOG_DEBUG("Converted split op into a list of IValues");
               return true;
             }})
        .pattern(
            {"aten::split_with_sizes(Tensor(a) self, int[] split_sizes, int dim=0) -> (Tensor[])",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               add_split(ctx, n, args, true, false);
               LOG_DEBUG("Converted split op into a list of IValues");
               return true;
             }})
        .pattern(
            {"aten::unbind.int(Tensor(a -> *) self, int dim=0) -> (Tensor[])",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               add_split(ctx, n, args, false, true);
               LOG_DEBUG("Converted split op into a list of IValues");
               return true;
             }})
        .pattern(
            {"aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> (Tensor)",
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
             }})
        .pattern(
            {"aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               int dim = args[1].unwrapToInt();
               auto index = args[2].ITensorOrFreeze(ctx);
               auto index_dim = index->getDimensions();
               std::vector<int64_t> dim_vec;
               for (int i = 0; i < index_dim.nbDims; i++) {
                 dim_vec.push_back(index_dim.d[i]);
               }
               auto value = args[3].unwrapToScalar() * torch::ones(dim_vec);
               auto value_tensor = tensor_to_const(ctx, value, "");
               if (self->getType() != value_tensor->getType()) {
                 value_tensor = castITensor(ctx, value_tensor, self->getType());
               }

               auto layer = ctx->net->addScatter(*self, *index, *value_tensor, nvinfer1::ScatterMode::kELEMENT);
               layer->setAxis(dim);

               TORCHTRT_CHECK(layer, "Unable to create layer for aten::scatter.value");

               layer->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               int dim = args[1].unwrapToInt();
               auto index = args[2].ITensorOrFreeze(ctx);
               auto src = args[3].ITensorOrFreeze(ctx);

               auto layer = ctx->net->addScatter(*self, *index, *src, nvinfer1::ScatterMode::kELEMENT);
               layer->setAxis(dim);

               TORCHTRT_CHECK(layer, "Unable to create layer for aten::scatter.src");

               layer->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], layer->getOutput(0));
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
