#include <ATen/ATen.h>
#include <vector>
#include "NvInfer.h"
#include "c10/util/intrusive_ptr.h"
#include "core/conversion/converters/converter_util.h"
#include "core/conversion/converters/converters.h"
#include "core/conversion/tensorcontainer/TensorContainer.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

// clang-format off
auto chunk_registrations TRTORCH_UNUSED =
    RegisterNodeConversionPatterns()
        .pattern({"aten::chunk(Tensor(a) self, int chunks, int dim=0) -> (Tensor[])",
                  [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                    auto in = args[0].ITensorOrFreeze(ctx);
                    auto numOutputs = args[1].unwrapToInt();
                    auto axis = args[2].unwrapToInt();
                    auto inDimSize = in->getDimensions().d[axis];
                    LOG_DEBUG("Number of chunk outputs: " << numOutputs);
                    std::vector<int64_t> sizes;

                    if (inDimSize % numOutputs == 0) {
                        for (int64_t i = 0; i < numOutputs; i++)
                            sizes.push_back(inDimSize / numOutputs);
                    } else {
                        for (int64_t i = 0; i < numOutputs - 1; i++)
                            sizes.push_back(inDimSize / numOutputs + 1);
                        sizes.push_back(inDimSize - (inDimSize / numOutputs + 1) * (numOutputs - 1));
                    }

                    c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
                    c10::TypePtr elementType = lt->getElementType();
                    auto list = c10::impl::GenericList(elementType);
                    list.reserve(numOutputs);

                    int start_idx = 0;
                    for (int64_t i = 0; i < numOutputs; i++) {
                        at::Tensor indices = torch::arange(start_idx, start_idx + sizes[i], 1).to(torch::kI32);
                        auto indicesTensor = tensor_to_const(ctx, indices);
                        auto gather_layer = ctx->net->addGather(*in, *indicesTensor, axis);
                        auto gather_out = gather_layer->getOutput(0);
                        auto tensor_holder = TensorContainer();
                        tensor_holder.hold_tensor(gather_out);
                        auto ival = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
                        list.emplace_back(ival);
                        start_idx = start_idx + sizes[i];
                    }

                    auto chunk_output_ivalue = std::move(torch::jit::IValue(list));
                    ctx->AssociateValueAndIValue(n->outputs()[0], chunk_output_ivalue);
                    LOG_DEBUG("Converted chunk op into a list of IValues");
                    return true;
                  }});
// clang-format on
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch