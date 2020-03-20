#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/constants.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/List.h"
#include "ATen/core/stack.h"

#include "core/conversion/evaluators/evaluators.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {
namespace {

auto prim_registrations = RegisterNodeEvaluators()
    .evaluator({
        torch::jit::prim::Constant,
        [](const torch::jit::Node* n, const kwargs& args) -> c10::optional<torch::jit::IValue> {
            if (n->output()->type()->kind() == at::FunctionType::Kind) {
                return {};
            }
            return torch::jit::toIValue(n->output());
        }
    }).evaluator({
        torch::jit::prim::ListConstruct,
        [](const torch::jit::Node* n, const kwargs& args) -> c10::optional<torch::jit::IValue> {
            const auto num_inputs = n->inputs().size();
            c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
            if (torch::jit::IntType::get() == lt->getElementType()) {
                c10::List<int64_t> list;
                list.reserve(num_inputs);
                for (auto in : n->inputs()) {
                    list.emplace_back(std::move(args.at(in)->to<int64_t>()));
                }
                return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            } else if (torch::jit::FloatType::get() == lt->getElementType()) {
                c10::List<double> list;
                list.reserve(num_inputs);
                for (auto in : n->inputs()) {
                    list.emplace_back(std::move(args.at(in)->to<double>()));
                }
                return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            } else if (lt->getElementType() == torch::jit::BoolType::get()) {
                c10::List<bool> list;
                list.reserve(num_inputs);
                for (auto in : n->inputs()) {
                    list.emplace_back(std::move(args.at(in)->to<bool>()));
                }
                return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            } else if (lt->getElementType()->isSubtypeOf(torch::jit::TensorType::get())) {
                c10::List<at::Tensor> list;
                list.reserve(num_inputs);
                for (auto in : n->inputs()) {
                    list.emplace_back(std::move(args.at(in)->toTensor()));
                }
                return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            } else {
                c10::TypePtr elementType = lt->getElementType();
                auto list = c10::impl::GenericList(elementType);
                list.reserve(num_inputs);
                for (auto in : n->inputs()) {
                    list.emplace_back(std::move(*(args.at(in))));
                }
                return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            }
        }
    });
}
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
