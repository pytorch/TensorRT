#include <bitset>
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto reduce_registrations TORCHTRT_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern(
            {"aten::mean(Tensor self, *, ScalarType? dtype=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto in_dims = util::toVec(in_tensor->getDimensions());
               LOG_WARNING("Mean Converter disregards dtype");

               uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);

               auto mean_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kAVG, axis_mask, false);

               TORCHTRT_CHECK(mean_layer, "Unable to create mean layer from node: " << *n);

               mean_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mean_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::mean.dim(Tensor self, int[] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto dims = args[1].unwrapToIntList();
               LOG_DEBUG("Dim to reduce:" << util::toDims(dims)); // Some abuse of toDim but just for debug info
               c10::List<int64_t> calculated_dims;
               auto in_dims = util::toVec(in_tensor->getDimensions());
               LOG_DEBUG("InDims " << in_dims); // Some abuse of toDim but just for debug info
               LOG_DEBUG(
                   "Dim to reduce (original): " << util::toDims(dims)); // Some abuse of toDim but just for debug info
               for (size_t i = 0; i < dims.size(); i++) {
                 auto dim_val = dims[i] < 0 ? (in_dims.size() + dims[i]) : dims[i];
                 calculated_dims.push_back(dim_val);
               }
               LOG_DEBUG(
                   "Dim to reduce (converted): "
                   << util::toDims(calculated_dims)); // Some abuse of toDim but just for debug info

               uint32_t axis_mask = 0;
               for (size_t d = 0; d < calculated_dims.size(); d++) {
                 axis_mask |= 1 << calculated_dims[d];
               }
               LOG_DEBUG("Axis Mask: " << std::bitset<32>(axis_mask));

               auto keepdim = args[2].unwrapToBool();
               LOG_DEBUG("Keep dims: " << keepdim);
               LOG_WARNING("Mean converter disregards dtype");
               auto mean_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kAVG, axis_mask, keepdim);
               TORCHTRT_CHECK(mean_layer, "Unable to create mean layer from node: " << *n);
               mean_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mean_layer->getOutput(0));
               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto in_dims = util::toVec(in_tensor->getDimensions());
               LOG_WARNING("Sum Converter disregards dtype");

               uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);

               auto sum_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kSUM, axis_mask, false);

               TORCHTRT_CHECK(sum_layer, "Unable to create sum layer from node: " << *n);

               sum_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], sum_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto dims = args[1].unwrapToIntList();
               c10::List<int64_t> calculated_dims;
               auto in_dims = util::toVec(in_tensor->getDimensions());
               LOG_DEBUG("InDims " << in_dims); // Some abuse of toDim but just for debug info
               LOG_DEBUG(
                   "Dim to reduce(original):" << util::toDims(dims)); // Some abuse of toDim but just for debug info
               for (size_t i = 0; i < dims.size(); i++) {
                 auto dim_val = dims[i] < 0 ? (in_dims.size() + dims[i]) : dims[i];
                 calculated_dims.push_back(dim_val);
               }

               LOG_DEBUG(
                   "Dim to reduce(converted):"
                   << util::toDims(calculated_dims)); // Some abuse of toDim but just for debug info

               uint32_t axis_mask = 0;
               for (size_t d = 0; d < calculated_dims.size(); d++) {
                 axis_mask |= 1 << calculated_dims[d];
               }
               LOG_DEBUG("Axis Mask: " << std::bitset<32>(axis_mask));

               auto keepdim = args[2].unwrapToBool();
               LOG_DEBUG("Keep dims: " << keepdim);

               LOG_WARNING("Sum converter disregards dtype");
               auto sum_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kSUM, axis_mask, keepdim);

               TORCHTRT_CHECK(sum_layer, "Unable to create sum layer from node: " << *n);

               sum_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], sum_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto in_dims = util::toVec(in_tensor->getDimensions());
               LOG_WARNING("Prod Converter disregards dtype");

               uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);

               auto prod_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kPROD, axis_mask, false);

               TORCHTRT_CHECK(prod_layer, "Unable to create sum layer from node: " << *n);

               prod_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], prod_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto in_dims = in_tensor->getDimensions();
               auto dim = args[1].unwrapToInt();
               LOG_DEBUG("Dim to reduce (original): " << dim);
               dim = dim < 0 ? (in_dims.nbDims + dim) : dim;
               LOG_DEBUG("Dim to reduce (converted): " << dim);

               uint32_t axis_mask = 1 << dim;
               LOG_DEBUG("Axis Mask: " << std::bitset<32>(axis_mask));

               auto keepdim = args[2].unwrapToBool();
               LOG_DEBUG("Keep dims: " << keepdim);

               LOG_WARNING("Prod converter disregards dtype");
               auto prod_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kPROD, axis_mask, keepdim);

               TORCHTRT_CHECK(prod_layer, "Unable to create mean layer from node: " << *n);

               prod_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], prod_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::max(Tensor self) -> Tensor",
             [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto in_dims = util::toVec(in_tensor->getDimensions());

               uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);

               auto max_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kMAX, axis_mask, false);

               TORCHTRT_CHECK(max_layer, "Unable to create max layer from node: " << *n);

               max_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], max_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }})
        .pattern(
            {"aten::min(Tensor self) -> Tensor", [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
               auto in_tensor = args[0].ITensorOrFreeze(ctx);
               auto in_dims = util::toVec(in_tensor->getDimensions());

               uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);

               auto min_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kMIN, axis_mask, false);

               TORCHTRT_CHECK(min_layer, "Unable to create min layer from node: " << *n);

               min_layer->setName(util::node_info(n).c_str());
               auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], min_layer->getOutput(0));

               LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
               return true;
             }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
