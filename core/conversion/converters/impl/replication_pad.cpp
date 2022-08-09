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

bool replication_padXd(ConversionCtx* ctx, const torch::jit::Node* n, args& args, int x_dim) {
  auto in = args[0].ITensor();
  auto inDims = in->getDimensions();
  int64_t inRank = inDims.nbDims;
  auto padding = args[1].unwrapToIntList().vec();
  if (padding.size() == 1) {
    for (int64_t i = 0; i < x_dim * 2 - 1; i++)
      padding.push_back(padding[0]);
  }
  if (inRank == 3) {
    TORCHTRT_CHECK(padding.size() == 2, "3D tensors expect 2 values for padding");
  } else if (inRank == 4) {
    TORCHTRT_CHECK(padding.size() == 4, "4D tensors expect 4 values for padding");
  } else if (inRank == 5) {
    TORCHTRT_CHECK(padding.size() == 6, "5D tensors expect 6 values for padding");
  } else {
    TORCHTRT_THROW_ERROR("Only 3D, 4D, 5D padding with non-constant padding are supported for now");
  }

  std::vector<nvinfer1::ITensor*> tensors_vec;
  // input: (N, C, D_in, H_in, W_in).
  // padding: (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
  // When axis is inRank - 1, making W_out = W_in + padding_left + padding_right.
  // When axis is inRank - 2, making H_out = H_in + padding_top + padding_bottom.
  // When axis is inRank - 1, making D_out = D_in + padding_front + padding_back.
  for (int64_t i = 0; i < int(padding.size() / 2); i++) {
    int64_t axis = inRank - (i + 1); // axis = {inRank - 1, inRank - 2, inRank - 3}
    int64_t padding_index = i * 2;

    if (padding[padding_index] > 0) { // left/top/front padding value
      tensors_vec.clear();
      at::Tensor left_indices = torch::tensor({0}, torch::kInt32);
      auto indicesTensor = tensor_to_const(ctx, left_indices);
      auto left_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
      auto left_gather_out = left_gather_layer->getOutput(0);
      for (int i = 0; i < padding[padding_index]; i++) {
        tensors_vec.push_back(left_gather_out);
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
        indicesTensor =
            ctx->net->addElementWise(*indicesTensor, *oneTensor, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
      } else {
        auto indices = torch::tensor({inDims.d[axis] - 1}, torch::kInt32);
        indicesTensor = tensor_to_const(ctx, indices);
      }
      auto right_gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
      auto right_gather_out = right_gather_layer->getOutput(0);

      for (int i = 0; i < padding[padding_index + 1]; i++) {
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
}

auto replication_pad_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::replication_pad1d(Tensor self, int[2] padding) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               replication_padXd(ctx, n, args, 1);
               return true;
             }})
        .pattern(
            {"aten::replication_pad2d(Tensor self, int[4] padding) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               replication_padXd(ctx, n, args, 2);
               return true;
             }})
        .pattern(
            {"aten::replication_pad3d(Tensor self, int[6] padding) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               replication_padXd(ctx, n, args, 3);
               return true;
             }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
