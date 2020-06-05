#include <limits>

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

auto prim_registrations = RegisterNodeEvaluators()
    .evaluator({
        torch::jit::prim::Constant,
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            if (n->output()->type()->kind() == at::FunctionType::Kind) {
                return {};
            }
            return torch::jit::toIValue(n->output());
        }
    }).evaluator({
        torch::jit::prim::NumToTensor,
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            return at::scalar_to_tensor(args.at(n->output(0)).IValue()->toScalar());
        }
    }).evaluator({
        torch::jit::prim::ListConstruct,
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            const auto num_inputs = n->inputs().size();
            if (constTypesOnly(args)) {
                c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
                if (torch::jit::IntType::get() == lt->getElementType()) {
                    c10::List<int64_t> list;
                    list.reserve(num_inputs);
                    for (auto in : n->inputs()) {
                        list.emplace_back(std::move(args.at(in).unwrapToInt()));
                    }
                    return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
                } else if (torch::jit::FloatType::get() == lt->getElementType()) {
                    c10::List<double> list;
                    list.reserve(num_inputs);
                    for (auto in : n->inputs()) {
                        list.emplace_back(std::move(args.at(in).unwrapToDouble()));
                    }
                    return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
                } else if (lt->getElementType() == torch::jit::BoolType::get()) {
                    c10::List<bool> list;
                    list.reserve(num_inputs);
                    for (auto in : n->inputs()) {
                        list.emplace_back(std::move(args.at(in).unwrapToBool()));
                    }
                    return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
                } else if (lt->getElementType()->isSubtypeOf(torch::jit::TensorType::get())) {
                    c10::List<at::Tensor> list;
                    list.reserve(num_inputs);
                    for (auto in : n->inputs()) {
                        if (args.at(in).isIValue()) {
                            list.emplace_back(std::move(args.at(in).unwrapToTensor()));
                        }
                    }
                    return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
                } else {
                    c10::TypePtr elementType = lt->getElementType();
                    auto list = c10::impl::GenericList(elementType);
                    list.reserve(num_inputs);
                    for (auto in : n->inputs()) {
                        list.emplace_back(std::move(*(args.at(in).IValue())));
                    }
                    return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
                }
            } else {
                c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
                c10::TypePtr elementType = lt->getElementType();
                auto list = c10::impl::GenericList(elementType);
                list.reserve(num_inputs);
                for (auto in : n->inputs()) {
                    if (args.at(in).isITensor()) {
                        auto tensor_holder = TensorContainer();
                        tensor_holder.hold_tensor(args.at(in).ITensor());
                        auto ival = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
                        list.emplace_back(std::move(ival));
                    } else {
                        list.emplace_back(std::move(args.at(in).unwrapToTensor()));
                    }
                }
                return c10::optional<torch::jit::IValue>(std::move(torch::jit::IValue(list)));
            }
        }
    }).evaluator({
         c10::Symbol::fromQualString("prim::min"),
         [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            auto a = args.at(n->input(0)).unwrapToIntList();
            int64_t min = std::numeric_limits<int64_t>::max();

            for (size_t i = 0; i < a.size(); i++) {
                if (a[i] < min) {
                    min = a[i];
                }
            }

            return min;
        },
        EvalOptions().validSchemas({"prim::min.self_int(int[] self) -> (int)"})
    }).evaluator({
        c10::Symbol::fromQualString("prim::shape"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            LOG_WARNING("There may be undefined behavior using dynamic shape and prim::shape");
            auto tensor_var = args.at(n->input(0));
            if (tensor_var.isITensor()) {
                auto tensor = tensor_var.ITensor();
                return util::toVec(tensor->getDimensions());
            } else {
                auto tensor = tensor_var.unwrapToTensor();
                return tensor.sizes();
            }
        },
        EvalOptions().validSchemas({
            "prim::shape(Tensor a) -> (int[])"
        })
    });
}
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
