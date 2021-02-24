#include "NvInfer.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "core/util/trt_util.h"
#include "plugins/cumsum_plugin.h"
#include "torch/torch.h"

#include <ATen/ATen.h>
#include <vector>

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

void create_plugin(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* in, const char* name, int dim) {
  LOG_WARNING("Cumsum layer will be run through ATen, not TensorRT. Performance may be lower than expected");

  auto creator = new plugins::CumsumPluginCreator();
  auto plugin = creator->createPlugin(name, dim);

  auto cumsum_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
  TRTORCH_CHECK(cumsum_layer, "Unable to create cumsum plugin from node" << *n);

  cumsum_layer->setName(util::node_info(n).c_str());

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], cumsum_layer->getOutput(0));

  LOG_DEBUG("Output tensor shape: " << layer_output->getDimensions());
}

auto cumsum_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::cumsum(Tensor self, int dim, *, int? dtype=None) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto input_dims = in->getDimensions();
       int dim = args[1].unwrapToInt();
       TRTORCH_CHECK(
           (dim >= 0 && dim < input_dims.nbDims) || (dim < 0 && (input_dims.nbDims + dim >= 0)),
           "Dimension out of range (expected to be in range of [" << -input_dims.nbDims << ", " << input_dims.nbDims - 1
                                                                  << "], but got " << dim << ")");
       if (dim < 0) {
         dim += input_dims.nbDims;
       }
       create_plugin(ctx, n, in, "Cumsum", dim);
       return true;
     }});

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
