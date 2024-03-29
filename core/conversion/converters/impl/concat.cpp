#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto cat_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns()
  .pattern({"aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
              auto ts = args[0].IValue()->toListRef();
              auto dim = args[1].unwrapToInt();

              std::vector<nvinfer1::ITensor*> tensors;
              for (auto t : ts) {
                if (t.isTensor()) {
                  auto torch_tensor = t.toTensor();
                  tensors.push_back(tensor_to_const(ctx, torch_tensor));
                } else {
                  auto cont = t.toCustomClass<TensorContainer>();
                  tensors.push_back(cont->tensor());
                }
              }

              auto promo_dtype = tensors[0]->getType();
              for(size_t idx = 1UL; idx < tensors.size(); ++idx){
                promo_dtype = promote_types(promo_dtype, tensors[idx]->getType());
              }

              for(size_t idx = 0UL; idx < tensors.size(); ++idx){
                if(tensors[idx]->getType() != promo_dtype){
                  tensors[idx] = castITensor(ctx, tensors[idx], promo_dtype, util::node_info(n) + "_cast_" + std::to_string(idx));
                }
              }

              if (dim < 0) {
                dim = tensors[0]->getDimensions().nbDims + dim;
              }

              auto cat_layer = ctx->net->addConcatenation(tensors.data(), tensors.size());
              cat_layer->setAxis(static_cast<int>(dim));
              auto cat_out = ctx->AssociateValueAndTensor(n->outputs()[0], cat_layer->getOutput(0));

              LOG_DEBUG("Output tensor shape: " << cat_out->getDimensions());

              return true;
            }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
