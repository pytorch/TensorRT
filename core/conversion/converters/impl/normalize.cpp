#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "plugins/normalize_plugin.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

/*
 * Helper functions
 */
void create_plugin(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* in,
    int64_t order,
    std::vector<int64_t> axes,
    bool keep_dims,
    const char* name) {
  LOG_WARNING("Normalize layer will be run through ATen, not TensorRT. Performance may be lower than expected");

  auto creator = new plugins::NormalizePluginCreator();
  auto inputnbDims = in->getDimensions().nbDims;
  for (int64_t i = 0; i < axes.size(); i++) {
    if (axes[i] < 0) {
      axes[i] += inputnbDims;
    }
    if (axes[i] > inputnbDims - 1) {
      TRTORCH_THROW_ERROR("Axis of normalization layer cannot exceed input rank");
    }
  }

  auto plugin = creator->createPlugin(name, order, axes, keep_dims);

  auto normalize_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
  TRTORCH_CHECK(normalize_layer, "Unable to create normalization plugin from node" << *n);

  normalize_layer->setName(util::node_info(n).c_str());

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], normalize_layer->getOutput(0));

  LOG_DEBUG("Normalize layer output tensor shape: " << layer_output->getDimensions());
}

auto normalize_registrations TRTORCH_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto in_shape = util::toVec(in->getDimensions());
       auto order = args[1].unwrapToScalar().to<int64_t>();
       auto axes = args[2].unwrapToIntList().vec();
       auto keep_dims = args[3].unwrapToBool();
       LOG_DEBUG("Order of normalize_plugin: " << order);
       LOG_DEBUG("Axis: " << axes);
       LOG_DEBUG("keep_dims: " << keep_dims);
       create_plugin(ctx, n, in, order, axes, keep_dims, "Normalize");
       return true;
     }

    });

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
