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
