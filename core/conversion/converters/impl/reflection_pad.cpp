#include <ATen/ATen.h>
#include <vector>
#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto reflection_padXd TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::reflection_pad2d(Tensor self, int[4] padding) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto inDims = in->getDimensions();
               int64_t inRank = inDims.nbDims;
               auto padding = args[1].unwrapToIntList().vec();
               if (padding.size() == 1) {
                 for (int64_t i = 0; i < 3; i++)
                   padding.push_back(padding[0]);
               }
               if (inRank == 4) {
                 TORCHTRT_CHECK(padding.size() == 4, "4D tensors expect 4 values for padding");
               } else {
                 TORCHTRT_THROW_ERROR("Only 4D padding are supported for now");
               }

               std::vector<nvinfer1::ITensor*> tensors_vec;
               // 2d padding: (padding_left, padding_right, padding_top, padding_bottom)

               for (int64_t i = 0; i < int(padding.size() / 2); i++) {
                 int64_t axis = inRank - (i + 1); // axis = {inRank - 1, inRank - 2}
                 int64_t padding_index = i * 2;

                 if (padding[padding_index] > 0) { // left/top padding value
                   tensors_vec.clear();

                   for (int i = 0; i < padding[padding_index]; i++) {
                     at::Tensor left_indices = torch::tensor({padding[padding_index] - i}, torch::kInt32);
                     auto indicesTensor = tensor_to_const(ctx, left_indices);
                     auto left_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
                     auto left_gather_out = left_gather_layer->getOutput(0);
                     tensors_vec.push_back(left_gather_out);
                   }
                   tensors_vec.push_back(in);
                   auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
                   concat_layer->setAxis(axis);
                   in = concat_layer->getOutput(0);
                   inDims = in->getDimensions();
                 }

                 if (padding[padding_index + 1] > 0) { // right/bottom padding value
                   tensors_vec.clear();
                   tensors_vec.push_back(in);

                   for (int i = 0; i < padding[padding_index + 1]; i++) {
                     nvinfer1::ITensor* indicesTensor = NULL;
                     auto indices = torch::tensor({inDims.d[axis] - 1 - (i + 1)}, torch::kInt32);
                     indicesTensor = tensor_to_const(ctx, indices);
                     auto right_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
                     auto right_gather_out = right_gather_layer->getOutput(0);
                     tensors_vec.push_back(right_gather_out);
                   }

                   auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
                   concat_layer->setAxis(axis);
                   in = concat_layer->getOutput(0);
                   inDims = in->getDimensions();
                 }
               }
               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }})
        .pattern(
            {"aten::reflection_pad1d(Tensor self, int[2] padding) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in = args[0].ITensor();
               auto inDims = in->getDimensions();
               int64_t inRank = inDims.nbDims;
               auto padding = args[1].unwrapToIntList().vec();
               if (padding.size() == 1) {
                 for (int64_t i = 0; i < 1; i++)
                   padding.push_back(padding[0]);
               }

               std::vector<nvinfer1::ITensor*> tensors_vec;
               // 1d padding: (padding_left, padding_right)

               int64_t axis = inRank - 1;
               int64_t padding_index = 0;

               if (padding[padding_index] > 0) { // left padding value
                 tensors_vec.clear();

                 for (int i = 0; i < padding[padding_index]; i++) {
                   at::Tensor left_indices = torch::tensor({padding[padding_index] - i}, torch::kInt32);
                   auto indicesTensor = tensor_to_const(ctx, left_indices);
                   auto left_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
                   auto left_gather_out = left_gather_layer->getOutput(0);
                   tensors_vec.push_back(left_gather_out);
                 }
                 tensors_vec.push_back(in);
                 auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
                 concat_layer->setAxis(axis);
                 in = concat_layer->getOutput(0);
                 inDims = in->getDimensions();
               }

               if (padding[padding_index + 1] > 0) { // right padding value
                 tensors_vec.clear();
                 tensors_vec.push_back(in);

                 for (int i = 0; i < padding[padding_index + 1]; i++) {
                   nvinfer1::ITensor* indicesTensor = NULL;
                   auto indices = torch::tensor({inDims.d[axis] - 1 - (i + 1)}, torch::kInt32);
                   indicesTensor = tensor_to_const(ctx, indices);
                   auto right_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
                   auto right_gather_out = right_gather_layer->getOutput(0);
                   tensors_vec.push_back(right_gather_out);
                 }

                 auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
                 concat_layer->setAxis(axis);
                 in = concat_layer->getOutput(0);
                 inDims = in->getDimensions();
               }

               auto out = ctx->AssociateValueAndTensor(n->outputs()[0], in);
               LOG_DEBUG("Output tensor shape: " << out->getDimensions());

               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
