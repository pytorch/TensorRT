#include "torch/torch.h"
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

#include <csignal>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto interpolate_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "aten::upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node*n, args& args) -> bool {
            TRTORCH_ASSERT(args[0].IValue()->isTensor(), "Input expected to be of type Tensor");

            auto in = args[0].ITensor();
            auto in_shape = util::toVec(in->getDimensions());

            // Case 1: user uses output size and not scales
            if (!args[1].IValue()->isNone() && args[2].IValue()->isNone()) {
                auto output_size = util::toDims(args[1].unwrapToIntList());

                TRTORCH_ASSERT(output_size.nbDims == 1, "aten::upsample_nearest1d input Tensor and output size dimension mismatch");
            } else {
                LOG_DEBUG("scale factor parameters not supported yet.");
            }

            return true;
        }
    }).pattern({
        "aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
//            std::raise(SIGINT);
            TRTORCH_ASSERT(args[0].IValue()->isTensor(), "Input expected to be of type Tensor");

            auto in = args[0].ITensor();
            auto in_shape = util::toVec(in->getDimensions());

            // Case 1: user uses output_size and not scales_h, scales_w
            if (!args[1].IValue()->isNone() && args[2].IValue()->isNone() && args[3].IValue()->isNone()){
                auto output_size = util::toDims(args[1].unwrapToIntList());

                TRTORCH_ASSERT( (output_size.nbDims == 1 || output_size.nbDims == 2), "aten::upsample_nearest2d input Tensor and output size dimension mismatch");

                nvinfer1::ILayer* new_layer;


        
                //util::toDims(args[1].unwrapToIntList());

            } else {
                LOG_DEBUG("scale factor parameters not supported yet.");
            }

            return true;
        }
    }).pattern({
        "aten::upsample_nearest3d(Tensor self, int[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> (Tensor)",
        [](ConversionCtx* ctx, const torch::jit::Node*n, args& args) -> bool {
            TRTORCH_ASSERT(args[0].IValue()->isTensor(), "Input expected to be of type Tensor");

            auto in = args[0].ITensor();
            auto in_shape = util::toVec(in->getDimensions());

            // Case 1: user uses output size and not scales_d, scales_h, scales_w
            if (!args[1].IValue()->isNone() && args[2].IValue()->isNone() && args[3].IValue()->isNone() && args[4].IValue()->isNone()) {
                auto output_size = util::toDims(args[1].unwrapToIntList());

                TRTORCH_ASSERT( (output_size.nbDims == 1 || output_size.nbDims == 3), "aten::upsample_nearest3d input Tensor and output size dimension mismatch");
                

            } else {
                LOG_DEBUG("scale factor parameters not supported yet.");
            }

            return true;
        }
    });


} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
