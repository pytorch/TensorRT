#include "torch/csrc/jit/ir/ir.h"
#include "torch/csrc/jit/ir/constants.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/List.h"
#include "ATen/core/stack.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/torch.h"

#include "core/conversion/evaluators/evaluators.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {
namespace {

auto aten_registrations = RegisterNodeEvaluators()
    .evaluator({
        c10::Symbol::fromQualString("aten::zeros"),
        // aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            auto options = torch::TensorOptions()
                .dtype(c10::ScalarType(args.at(&(n->output()[1])).unwrapToInt()))
                .layout(torch::kStrided)
                .device(torch::kCUDA);

            auto out_tensor = torch::zeros(args.at(&(n->output()[0])).unwrapToIntList().vec(), options);
            return out_tensor;
        }
    });
}
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
