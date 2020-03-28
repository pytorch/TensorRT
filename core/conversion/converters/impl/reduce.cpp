#include <bitset>
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {
auto reduced_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::mean(Tensor self, *, ScalarType? dtype=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in_tensor = args[0].ITensor();
            auto in_dims = util::toVec(in_tensor->getDimensions());
            LOG_WARNING("Mean Converter disregards dtype");

            uint32_t axis_mask = (uint32_t)(((uint64_t)1 << in_dims.size()) - 1);

            auto mean_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kAVG, axis_mask, false);

            TRTORCH_CHECK(mean_layer, "Unable to create mean layer from node: " << *n);

            mean_layer->setName(util::node_info(n).c_str());
            auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mean_layer->getOutput(0));

            LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
            return true;
        }
    }).pattern({
        "aten::mean.dim(Tensor self, int[] dim, bool keepdim=False, *, int? dtype=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            auto in_tensor = args[0].ITensor();
            auto dims = args[1].unwrapToIntList();
            LOG_DEBUG("Dim to reduce:" << util::toDims(dims)); // Some abuse of toDim but just for debug info

            uint32_t axis_mask = 0;
            for (int d = 0; d < dims.size(); d++) {
                axis_mask |= 1 << dims[d];
            }
            LOG_DEBUG("Axis Mask" << std::bitset<32>(axis_mask));

            auto keepdim = args[2].unwrapToBool();
            LOG_DEBUG("Keep dims :" << keepdim);

            LOG_WARNING("Mean converter disregards dtype");
            auto mean_layer = ctx->net->addReduce(*in_tensor, nvinfer1::ReduceOperation::kAVG, axis_mask, keepdim);

            TRTORCH_CHECK(mean_layer, "Unable to create mean layer from node: " << *n);

            mean_layer->setName(util::node_info(n).c_str());
            auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], mean_layer->getOutput(0));

            LOG_DEBUG("Output shape: " << out_tensor->getDimensions());
            return true;
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch

// #include "core/util/prelude.h"
// #include "core/conversion/converters/converters.h"

// namespace trtorch {
// namespace core {
// namespace conversion {
// namespace converters {
// namespace impl {
// namespace {

// #define convert(unary, trt_type)                                                \
//   auto unary##_registrations TRTORCH_UNUSED =                                   \
//       RegisterNodeConversionPatterns().pattern(                                 \
//           {"aten::" #unary "(Tensor self) -> Tensor",                           \
//            [](ConversionCtx *ctx, const torch::jit::Node *n,                    \
//               args &args) -> bool {                                             \
//              auto in = args[0].ITensor();                                       \
//              auto unary =                                                       \
//                  ctx->net->addUnary(*in, nvinfer1::UnaryOperation::trt_type);   \
//                                                                                 \
//              TRTORCH_CHECK(                                                     \
//                  unary,                                                         \
//                  "Unable to create " #unary " layer from node: " << *n);        \
//                                                                                 \
//              unary->setName(util::node_info(n).c_str());                        \
//              auto out_tensor = ctx->AssociateValueAndTensor(                 \
//                                                           n->outputs()[0],      \
//                                                           unary->getOutput(0)); \
//              LOG_DEBUG(                                                         \
//                  "Output tensor shape: " << out_tensor->getDimensions());       \
//                                                                                 \
//              return true;                                                       \
//            }});

// convert(cos, kCOS);
// convert(acos, kACOS);
// convert(cosh, kCOSH);
// convert(sin, kSIN);
// convert(asin, kASIN);
// convert(sinh, kSINH);
// convert(tan, kTAN);
// convert(atan, kATAN);
// convert(abs, kABS);
// convert(floor, kFLOOR);
// convert(reciprocal, kRECIP);
// convert(log, kLOG);
// convert(ceil, kCEIL);
// convert(sqrt, kSQRT);
// convert(exp, kEXP);
// convert(neg, kNEG);

// #undef convert

// } // namespace
// } // namespace impl
// } // namespace converters
// } // namespace conversion
// } // namespace core
// } // namespace trtorch
