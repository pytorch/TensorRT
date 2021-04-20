#include <ATen/ATen.h>
#include <vector>
#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto replication_pad_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto inDims = in->getDimensions();
       int64_t inRank = inDims.nbDims;
       auto padding = args[1].unwrapToIntList().vec();
       int64_t padSize = padding.size();
       auto value = args[2].unwrapToScalar().to<float>();

       TRTORCH_CHECK(padSize % 2 == 0, "Length of pad must be even but instead it equals " << padSize);

       int64_t l_pad = padSize / 2;
       TRTORCH_CHECK(
           inRank >= (int64_t)l_pad,
           "Length of pad should be no more than twice the number of "
           "dimensions of the input. Pad length is "
               << padSize << "while the input has " << inRank << "dimensions.");

       // TODO negative padding

       std::vector<nvinfer1::ITensor*> tensors_vec;
       // input: (N, C, D_in, H_in, W_in).
       // padding: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
       // When axis is inRank - 1, making W_out = W_in + padding_left + padding_right.
       // When axis is inRank - 2, making H_out = H_in + padding_top + padding_bottom.
       // When axis is inRank - 3, making D_out = D_in + padding_front + padding_back.
       for (int64_t i = 0; i < l_pad; i++) {
         int64_t axis = inRank - (i + 1); // axis = {inRank - 1, inRank - 2, inRank - 3}
         int64_t padding_index = i * 2;

         if (padding[padding_index] > 0) { // left/top/front padding value
           tensors_vec.clear();
           at::Tensor left_indices = torch::tensor({0}, torch::kInt32);
           auto indicesTensor = tensor_to_const(ctx, left_indices);
           auto left_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
           auto left_gather_out = left_gather_layer->getOutput(0);

           // fill the left_gather_out with value
           auto fill_layer = ctx->net->addFill(nvinfer1::Dims{1, {1}}, nvinfer1::FillOperation::kLINSPACE);
           auto shape_gather_out = ctx->net->addShape(*left_gather_out)->getOutput(0);
           fill_layer->setInput(0, *shape_gather_out);
           at::Tensor value_tensor = torch::tensor(value, torch::kFloat32);
           auto valueTensor = tensor_to_const(ctx, value_tensor);
           fill_layer->setInput(1, *valueTensor);
           at::Tensor delta_tensor = torch::zeros(inRank);
           auto deltaTensor = tensor_to_const(ctx, delta_tensor);
           fill_layer->setInput(2, *deltaTensor);
           auto padTensor = fill_layer->getOutput(0);

           for (int i = 0; i < padding[padding_index]; i++) {
             tensors_vec.push_back(padTensor);
           }
           tensors_vec.push_back(in);
           auto concat_layer = ctx->net->addConcatenation(tensors_vec.data(), tensors_vec.size());
           concat_layer->setAxis(axis);
           in = concat_layer->getOutput(0);
           inDims = in->getDimensions();
         }

         if (padding[padding_index + 1] > 0) { // right/bottom/back padding value
           tensors_vec.clear();
           tensors_vec.push_back(in);

           nvinfer1::ITensor* indicesTensor = NULL;
           if (inDims.d[axis] == -1) {
             auto shapeTensor = ctx->net->addShape(*in)->getOutput(0);
             at::Tensor dimValue = torch::tensor({axis}, torch::kInt32);
             auto dimTensor = tensor_to_const(ctx, dimValue);
             indicesTensor = ctx->net->addGather(*shapeTensor, *dimTensor, 0)->getOutput(0);
             auto oneTensor = tensor_to_const(ctx, torch::tensor({1}, torch::kInt32));
             indicesTensor = ctx->net->addElementWise(*indicesTensor, *oneTensor, nvinfer1::ElementWiseOperation::kSUB)
                                 ->getOutput(0);
           } else {
             auto indices = torch::tensor({inDims.d[axis] - 1}, torch::kInt32);
             indicesTensor = tensor_to_const(ctx, indices);
           }
           auto right_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
           auto right_gather_out = right_gather_layer->getOutput(0);

           // fill the right_gather_out with value
           auto fill_layer = ctx->net->addFill(nvinfer1::Dims{1, {1}}, nvinfer1::FillOperation::kLINSPACE);
           auto shape_gather_out = ctx->net->addShape(*right_gather_out)->getOutput(0);
           fill_layer->setInput(0, *shape_gather_out);
           at::Tensor value_tensor = torch::tensor(value, torch::kFloat32);
           auto valueTensor = tensor_to_const(ctx, value_tensor);
           fill_layer->setInput(1, *valueTensor);
           at::Tensor delta_tensor = torch::zeros(inRank);
           auto deltaTensor = tensor_to_const(ctx, delta_tensor);
           fill_layer->setInput(2, *deltaTensor);
           auto padTensor = fill_layer->getOutput(0);

           for (int i = 0; i < padding[padding_index + 1]; i++) {
             tensors_vec.push_back(padTensor);
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
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch