#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
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
    std::vector<int32_t> axes,
    bool keep_dims,
    const char* name) {
  LOG_WARNING("Normalize layer will be run through ATen, not TensorRT. Performance may be lower than expected");
  nvinfer1::PluginFieldCollection fc;
  std::vector<nvinfer1::PluginField> f;
  f.emplace_back(nvinfer1::PluginField("order", &order, nvinfer1::PluginFieldType::kINT32, 1));
  f.emplace_back(nvinfer1::PluginField("axes", axes.data(), nvinfer1::PluginFieldType::kINT32, axes.size()));
  f.emplace_back(nvinfer1::PluginField("keep_dims", &keep_dims, nvinfer1::PluginFieldType::kINT32, 1));
  fc.nbFields = f.size();
  fc.fields = f.data();

  auto inputnbDims = in->getDimensions().nbDims;
  for (int64_t i = 0; i < (int64_t)axes.size(); i++) {
    if (axes[i] < 0) {
      axes[i] += inputnbDims;
    }
    if (axes[i] > inputnbDims - 1) {
      TORCHTRT_THROW_ERROR("Axis of normalization layer cannot exceed input rank");
    }
  }

  auto creator = getPluginRegistry()->getPluginCreator("NormalizePlugin", "1", "torch_tensorrt");
  auto plugin = creator->createPlugin(name, &fc);
  auto normalize_layer = ctx->net->addPluginV2(reinterpret_cast<nvinfer1::ITensor* const*>(&in), 1, *plugin);
  TORCHTRT_CHECK(normalize_layer, "Unable to create normalization plugin from node" << *n);

  normalize_layer->setName(util::node_info(n).c_str());

  auto layer_output = ctx->AssociateValueAndTensor(n->outputs()[0], normalize_layer->getOutput(0));

  LOG_DEBUG("Normalize layer output tensor shape: " << layer_output->getDimensions());
}

auto normalize_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> (Tensor)",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensor();
       auto in_shape = util::toVec(in->getDimensions());
       auto order = args[1].unwrapToScalar().to<int32_t>();
       auto axes_values = args[2].unwrapToIntList().vec();
       std::vector<int32_t> axes(axes_values.begin(), axes_values.end());
       auto keep_dims = (int32_t)args[3].unwrapToBool();
       LOG_DEBUG("Order of normalize_plugin: " << order);
       LOG_DEBUG("Axis: " << axes);
       LOG_DEBUG("keep_dims: " << keep_dims);
       create_plugin(ctx, n, in, order, axes, keep_dims, "NormalizePluginTorchTRT");
       return true;
     }

    });

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
