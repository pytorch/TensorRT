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

auto constant_pad_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto in_dims = in->getDimensions();
       int64_t in_rank = in_dims.nbDims;
       auto padding = args[1].unwrapToIntList().vec();
       int64_t pad_size = padding.size();
       auto value = args[2].unwrapToScalar().to<float>();
       at::Tensor value_tensor = torch::tensor(value, util::TRTDataTypeToScalarType(in->getType()));
       auto value_itensor = tensor_to_const(ctx, value_tensor);
       TORCHTRT_CHECK(pad_size % 2 == 0, "Length of pad must be even but instead it equals " << pad_size);

       std::vector<int64_t> start(in_rank, 0);
       std::vector<int64_t> total_padding(in_rank, 0);
       std::vector<int64_t> stride(in_rank, 1);

       // Padding is stored (left, right) starting from the last dim and working backwards
       for (size_t i = 0UL; i < padding.size(); i += 2) {
         auto left = padding[i];
         TORCHTRT_CHECK(left >= 0, "Unsupported negative pad at index " << i);
         auto right = padding[i + 1];
         TORCHTRT_CHECK(right >= 0, "Unsupported negative pad at index " << i + 1);
         auto idx = in_rank - ((i / 2) + 1);
         start[idx] = -left;
         total_padding[idx] = left + right;
       }

       auto size = stride; // placeholder for the dynamic case
       if (!ctx->input_is_dynamic) {
         size = total_padding;
         for (size_t i = 0UL; i < total_padding.size(); ++i) {
           size[i] += in_dims.d[i];
         }
       }

       auto slice_layer = ctx->net->addSlice(
           *in,
           util::toDims(c10::IntArrayRef(start)),
           util::toDims(c10::IntArrayRef(size)),
           util::toDims(c10::IntArrayRef(stride)));
       TORCHTRT_CHECK(slice_layer, "Unable to create slice layer from node: " << *n);
       slice_layer->setName((util::node_info(n) + "_slice").c_str());
       slice_layer->setMode(nvinfer1::SampleMode::kFILL);
       slice_layer->setInput(4, *value_itensor);

       if (ctx->input_is_dynamic) {
         // build the size using inetwork layers
         auto total_padding_itensor = tensor_to_const(ctx, torch::tensor(total_padding, torch::kInt32));
         nvinfer1::ITensor* shapeOutput = getShapeOutput(ctx, in, (util::node_info(n) + "_shape").c_str());
         auto add_layer =
             ctx->net->addElementWise(*shapeOutput, *total_padding_itensor, nvinfer1::ElementWiseOperation::kSUM);
         TORCHTRT_CHECK(add_layer, "Unable to create add layer from node: " << *n);
         add_layer->setName((util::node_info(n) + "_add").c_str());
         slice_layer->setInput(2, *add_layer->getOutput(0));
       }

       auto out = ctx->AssociateValueAndTensor(n->outputs()[0], slice_layer->getOutput(0));
       LOG_DEBUG("Output tensor shape: " << out->getDimensions());
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
