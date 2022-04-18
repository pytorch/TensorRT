#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto cumsum_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::cumsum(Tensor self, int dim, *, int? dtype=None) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       auto input_dims = in->getDimensions();
       int dim = args[1].unwrapToInt();
       TORCHTRT_CHECK(
           (dim >= 0 && dim < input_dims.nbDims) || (dim < 0 && (input_dims.nbDims + dim >= 0)),
           "Dimension out of range (expected to be in range of [" << -input_dims.nbDims << ", " << input_dims.nbDims - 1
                                                                  << "], but got " << dim << ")");
       if (dim < 0) {
         dim += input_dims.nbDims;
       }

       // Scan through each slice across summation axis and add it to the running sum
       auto loop = ctx->net->addLoop();
       nvinfer1::ITensor* tripLimit = NULL;
       if (input_dims.d[dim] > 0) {
         torch::Tensor axis = torch::tensor(input_dims.d[dim], torch::kInt32);
         tripLimit = tensor_to_const(ctx, axis);
       } else {
         nvinfer1::ITensor* inpShape = ctx->net->addShape(*in)->getOutput(0);
         torch::Tensor dimValue = torch::tensor(dim, torch::kInt32);
         nvinfer1::ITensor* axis = tensor_to_const(ctx, dimValue);
         tripLimit = ctx->net->addGather(*inpShape, *axis, 0)->getOutput(0);
       }

       loop->addTripLimit(*tripLimit, nvinfer1::TripLimit::kCOUNT);

       auto iterator = loop->addIterator(*in, dim, false);
       auto data = iterator->getOutput(0);
       auto newDims = data->getDimensions();

       torch::Tensor zeroValue =
           at::full(util::toVec(newDims), 0, torch_tensorrt::core::util::TRTDataTypeToScalarType(in->getType()));
       auto zeroTensor = tensor_to_const(ctx, zeroValue);
       auto runningSum = loop->addRecurrence(*zeroTensor);
       auto runningSumTensor = runningSum->getOutput(0);

       auto curSum = ctx->net->addElementWise(*data, *runningSumTensor, nvinfer1::ElementWiseOperation::kSUM);
       runningSum->setInput(1, *curSum->getOutput(0));

       nvinfer1::ILoopOutputLayer* loopOut =
           loop->addLoopOutput(*curSum->getOutput(0), nvinfer1::LoopOutput::kCONCATENATE, dim);
       loopOut->setInput(1, *tripLimit);

       auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], loopOut->getOutput(0));

       LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
