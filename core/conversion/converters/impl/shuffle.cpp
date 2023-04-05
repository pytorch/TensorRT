#include "core/conversion/converters/converters.h"

#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

static auto shuffle_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto start_dim = args[1].unwrapToInt();
               auto end_dim = args[2].unwrapToInt();
               auto in_shape = util::toVec(in->getDimensions());
               std::vector<int64_t> out_shape;
               if (ctx->input_is_dynamic) {
                 end_dim = (end_dim == -1) ? in_shape.size() - 1 : end_dim;
                 int nbDynamicFlattenedDims = 0;
                 int nbDynamicUnflattenedDims = 0;
                 for (int i = 0; i < (int)in_shape.size(); i++) {
                   if (in_shape[i] == -1) {
                     if (i >= start_dim && i <= end_dim)
                       nbDynamicFlattenedDims++;
                     else
                       nbDynamicUnflattenedDims++;
                   }
                 }
                 if (nbDynamicFlattenedDims > 0 && nbDynamicUnflattenedDims > 0) {
                   TORCHTRT_THROW_ERROR(
                       "Flatten is currently not supported when target shape contains more than one dynamic dimension");
                 }
                 if (nbDynamicUnflattenedDims > 1) {
                   TORCHTRT_THROW_ERROR(
                       "Flatten is currently not supported when target shape contains more than one dynamic dimension");
                 }
                 out_shape = in_shape;
                 out_shape.erase(std::begin(out_shape) + start_dim, std::begin(out_shape) + end_dim + 1);
                 if (nbDynamicFlattenedDims == 0) {
                   auto flattened_dim = std::accumulate(
                       std::begin(in_shape) + start_dim,
                       std::begin(in_shape) + end_dim + 1,
                       1,
                       std::multiplies<int64_t>());
                   out_shape.insert(std::begin(out_shape) + start_dim, flattened_dim);
                 } else {
                   out_shape.insert(std::begin(out_shape) + start_dim, -1);
                 }
               } else {
                 out_shape = torch::flatten(torch::rand(in_shape), start_dim, end_dim).sizes().vec();
               }

               auto shuffle = ctx->net->addShuffle(*in);
               TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
               shuffle->setReshapeDimensions(util::toDims(out_shape));
               shuffle->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::unflatten.int(Tensor self, int dim, int[] sizes) -> (Tensor)",
              [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                auto in = args[0].ITensorOrFreeze(ctx);
                auto dim = args[1].unwrapToInt();
                auto in_shape = util::toVec(in->getDimensions());
                std::vector<int64_t> new_shape;
                nvinfer1::ITensor* shape_tensor;
                if (ctx->input_is_dynamic) {
                  /*
                   * In case the dim is negative
                   * If the dim in negative range is larger than in_shape,
                   * then it should run into index out of bound error as expected
                   */
                  if (dim < 0) {
                    dim = in_shape.size() + dim;
                  }
                  std::cout << "Dynamic shape case" << std::endl;
                  LOG_DEBUG("Using dynamic version of reshape layer");
                  if (args[2].isITensorList()) {
                    std::cout << "isTensorList case" << std::endl;
                    LOG_DEBUG("Shape tensor is an ITensorList");
                    auto expand_shape = args[2].unwrapToITensorList();
                    auto shape_layer = ctx->net->addShape(*in);
                    TORCHTRT_CHECK(shape_layer, "Unable to create shape layer from node: " << *n);
                    auto shape_1d_tensor = shape_layer->getOutput(0);

                    std::vector<int> before_dim_indices_vector(dim);
                    std::iota(before_dim_indices_vector.begin(), before_dim_indices_vector.end(), 0);

                    nvinfer1::ITensor* before_dim_gather_out = nullptr;
                    if(before_dim_indices_vector.size()){
                      at::Tensor before_dim_indices = torch::tensor(before_dim_indices_vector).to(torch::kI32);
                      auto before_dim_indices_out = converters::tensor_to_const(ctx, before_dim_indices);
                      auto before_dim_gather_layer = ctx->net->addGather(*shape_1d_tensor, *before_dim_indices_out, 0);
                      TORCHTRT_CHECK(before_dim_gather_layer, "Unable to create gather layer from node: " << *n);
                      before_dim_gather_out = before_dim_gather_layer->getOutput(0);
                    }

                    std::vector<int> after_dim_indices_vector(in_shape.size() - (dim + 1));
                    std::iota(after_dim_indices_vector.begin(), after_dim_indices_vector.end(), dim + 1);

                    nvinfer1::ITensor* after_dim_gather_out = nullptr;
                    if(after_dim_indices_vector.size()){ 
                      at::Tensor after_dim_indices = torch::tensor(after_dim_indices_vector).to(torch::kI32);
                      auto after_dim_indices_out = converters::tensor_to_const(ctx, after_dim_indices);
                      auto after_dim_gather_layer = ctx->net->addGather(*shape_1d_tensor, *after_dim_indices_out, 0);
                      TORCHTRT_CHECK(after_dim_gather_layer, "Unable to create gather layer from node: " << *n);
                      after_dim_gather_out = after_dim_gather_layer->getOutput(0);
                    }

                    std::vector<nvinfer1::ITensor*> shape_tensors;
                    if(before_dim_gather_out){
                      shape_tensors.push_back(before_dim_gather_out);
                    }
                    for(auto new_shape_tensor : expand_shape){
                      shape_tensors.push_back(new_shape_tensor);
                    }
                    if(after_dim_gather_out){
                      shape_tensors.push_back(after_dim_gather_out);
                    }

                    auto shape_cat_layer = ctx->net->addConcatenation(shape_tensors.data(), shape_tensors.size());
                    TORCHTRT_CHECK(shape_cat_layer, "Unable to create cat layer from node: " << *n);
                    shape_tensor = shape_cat_layer->getOutput(0);
                    LOG_DEBUG("Shape tensor shape: " << shape_tensor->getDimensions());
                  } else if (args[2].isIntList()) {
                    auto shape_vec = args[2].unwrapToIntList().vec();                    
                    // New shape 
                    new_shape.insert(new_shape.end(), in_shape.begin(), in_shape.begin() + dim);
                    new_shape.insert(new_shape.end(), shape_vec.begin(), shape_vec.end());
                    new_shape.insert(new_shape.end(), in_shape.begin() + dim + 1, in_shape.end());

                    shape_tensor = tensor_to_const(ctx, torch::tensor(new_shape).to(torch::kI32));
                  } else {
                    LOG_ERROR(
                      "Invalid IValue type of " <<  args[2].ivalue_type()
                                                << " detected for shape tensor from node: " << *n);
                  }
                }
                else {
                  new_shape = torch::unflatten(torch::rand(in_shape), dim, args[2].unwrapToIntList().vec()).sizes().vec();
                }
                auto shuffle = ctx->net->addShuffle(*in);
                shuffle->setName(util::node_info(n).c_str());
                TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);

                if (ctx->input_is_dynamic) {
                  shuffle->setInput(1, *shape_tensor);
                } else {
                  shuffle->setReshapeDimensions(util::toDims(new_shape));
                }

                auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
                LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

                return true;
              }})
        .pattern(
            {"aten::reshape(Tensor self, int[] shape) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto in_shape = util::toVec(in->getDimensions());
               std::vector<int64_t> new_shape;
               if (ctx->input_is_dynamic) {
                 new_shape = util::toVec(args[1].unwrapToIntList().vec());
                 int nbDynamicDims = 0;
                 for (size_t i = 0; i < new_shape.size(); i++) {
                   if (in_shape[i] == -1)
                     nbDynamicDims++;
                 }
                 if (nbDynamicDims > 1) {
                   TORCHTRT_THROW_ERROR(
                       "Resize is currently not supported when target shape contains more than one dynamic dimension");
                 }
               } else {
                 new_shape = torch::reshape(torch::rand(in_shape), args[1].unwrapToIntList().vec()).sizes().vec();
               }

               auto shuffle = ctx->net->addShuffle(*in);
               TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
               shuffle->setReshapeDimensions(util::toDims(new_shape));
               shuffle->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::view(Tensor(a) self, int[] size) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto in_shape = util::toVec(in->getDimensions());

               auto shuffle = ctx->net->addShuffle(*in);
               TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
               shuffle->setReshapeDimensions(util::toDims(args[1].unwrapToIntList().vec()));
               shuffle->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::permute(Tensor(a) self, int[] dims) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto in_shape = util::toVec(in->getDimensions());
               auto new_order = args[1].unwrapToIntList().vec();

               LOG_DEBUG("Shuffle to: " << util::toDims(new_order));

               auto shuffle = ctx->net->addShuffle(*in);
               TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
               nvinfer1::Permutation permute;
               std::copy(new_order.begin(), new_order.end(), permute.order);
               shuffle->setSecondTranspose(permute);
               shuffle->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> (Tensor(a))",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto in_shape = util::toVec(in->getDimensions());
               auto ndims = in_shape.size();
               auto dim0 = args[1].unwrapToInt();
               auto dim1 = args[2].unwrapToInt();

               std::vector<int64_t> new_order;
               for (size_t i = 0; i < ndims; i++) {
                 new_order.push_back(i);
               }
               dim0 = dim0 < 0 ? (dim0 + ndims) : dim0;
               dim1 = dim1 < 0 ? (dim1 + ndims) : dim1;
               auto tmp = dim0;
               new_order[dim0] = new_order[dim1];
               new_order[dim1] = tmp;

               LOG_DEBUG("Shuffle to: " << util::toDims(new_order));

               auto shuffle = ctx->net->addShuffle(*in);
               TORCHTRT_CHECK(shuffle, "Unable to create shuffle layer from node: " << *n);
               nvinfer1::Permutation permute;
               std::copy(new_order.begin(), new_order.end(), permute.order);

               shuffle->setSecondTranspose(permute);
               shuffle->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::t(Tensor self) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensorOrFreeze(ctx);
               auto input_dims = in->getDimensions();
               // For input tensors < 2D, return them as is
               // For a 2D input tensor, return transpose(input, 0, 1) which is a general 2d matrix transpose.
               if (input_dims.nbDims < 2) {
                 auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], in);
                 LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
                 return true;
               }

               auto shuffle_layer = ctx->net->addShuffle(*in);
               TORCHTRT_CHECK(shuffle_layer, "Unable to create shuffle layer from node: " << *n);
               nvinfer1::Permutation firstPerm;
               firstPerm.order[0] = 1;
               firstPerm.order[1] = 0;

               shuffle_layer->setFirstTranspose(firstPerm);
               shuffle_layer->setZeroIsPlaceholder(false);
               shuffle_layer->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], shuffle_layer->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::pixel_shuffle(Tensor self, int upscale_factor) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto self = args[0].ITensorOrFreeze(ctx);
               auto in_shape = util::toVec(self->getDimensions());
               int64_t irank = in_shape.size();
               TORCHTRT_CHECK(
                   irank >= 3,
                   "pixel_shuffle expects input to have at least 3 dimensions, but got input with " << irank
                                                                                                    << " dimension(s)");
               int64_t upscale_factor = args[1].unwrapToInt();
               TORCHTRT_CHECK(
                   upscale_factor > 0, "pixel_shuffle expects a positive upscale_factor, but got " << upscale_factor);
               int64_t upscale_factor_squared = upscale_factor * upscale_factor;

               const auto NUM_NON_BATCH_DIMS = 3;
               const auto self_sizes_batch_end = in_shape.end() - NUM_NON_BATCH_DIMS;

               int64_t ic = in_shape[irank - 3];
               int64_t ih = in_shape[irank - 2];
               int64_t iw = in_shape[irank - 1];

               TORCHTRT_CHECK(
                   ic % upscale_factor_squared == 0,
                   "pixel_shuffle expects its input's 'channel' dimension to be divisible by the square of "
                       << "upscale_factor, but input.size(-3)=" << ic << " is not divisible by "
                       << upscale_factor_squared);

               int64_t oc = ic / upscale_factor_squared;
               int64_t oh = ih * upscale_factor;
               int64_t ow = iw * upscale_factor;

               // First, reshape to split the channels dim from c into 3 separate dims: (oc,
               // upscale_factor, upscale_factor). This allows shuffling to be done next by
               // permuting dims.
               std::vector<int64_t> added_dims_shape(in_shape.begin(), self_sizes_batch_end);
               added_dims_shape.insert(added_dims_shape.end(), {oc, upscale_factor, upscale_factor, ih, iw});
               auto view_layer = ctx->net->addShuffle(*self);
               TORCHTRT_CHECK(view_layer, "Unable to create shuffle layer from node: " << *n);
               view_layer->setReshapeDimensions(util::toDims(added_dims_shape));
               int64_t view_rank = added_dims_shape.size();

               // Next, shuffle by permuting the new upscale_factor dims alongside the height and width dims.
               auto permutation_layer = ctx->net->addShuffle(*view_layer->getOutput(0));
               TORCHTRT_CHECK(permutation_layer, "Unable to create shuffle layer from node: " << *n);
               // std::iota is used to maintain the batch dims within the permutation.
               // Eg: if added_dims_shape is {n1, n2, c, r, r, h, w}, then the new_order is {view_rank-7,
               // view_rank-6, view_rank-5, view_rank-2, view_rank-4, view_rank-1, view_rank-3}
               std::vector<int64_t> new_order(in_shape.begin(), self_sizes_batch_end);
               std::iota(new_order.begin(), new_order.end(), 0);
               new_order.insert(
                   new_order.end(),
                   {view_rank - 5 /* oc */,
                    view_rank - 2 /* ih */,
                    view_rank - 4 /* 1st upscale_factor */,
                    view_rank - 1 /* iw */,
                    view_rank - 3 /* 2nd upscale_factor */});
               nvinfer1::Permutation permute;
               std::copy(new_order.begin(), new_order.end(), permute.order);
               permutation_layer->setSecondTranspose(permute);

               // Finally, upscale by collapsing (ih, upscale_factor) -> a single dim (oh)
               // and (iw, upscale_factor) -> a single dim (ow).
               std::vector<int64_t> final_shape(in_shape.begin(), self_sizes_batch_end);
               final_shape.insert(final_shape.end(), {oc, oh, ow});
               auto last_view_layer = ctx->net->addShuffle(*permutation_layer->getOutput(0));
               TORCHTRT_CHECK(last_view_layer, "Unable to create shuffle layer from node: " << *n);
               last_view_layer->setReshapeDimensions(util::toDims(final_shape));
               last_view_layer->setName(util::node_info(n).c_str());

               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], last_view_layer->getOutput(0));
               LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());

               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
