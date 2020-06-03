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


int64_t normalizeIndex(int64_t idx, int64_t list_size) {
    if (idx < 0) {
        // Handle negative indexing
        idx = list_size + idx;
    }
    return idx;
}

auto aten_registrations = RegisterNodeEvaluators()
    .evaluator({
        c10::Symbol::fromQualString("aten::zeros"),
        // aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            auto options = torch::TensorOptions()
                .dtype(c10::ScalarType(args.at(n->output(1)).unwrapToInt()))
                .layout(torch::kStrided)
                .device(torch::kCUDA);

            auto out_tensor = torch::zeros(args.at(n->input(0)).unwrapToIntList().vec(), options);
            return out_tensor;
        }
    }).evaluator({
        c10::Symbol::fromQualString("aten::mul"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            auto a = args.at(n->input(0)).unwrapToInt();
            auto b = args.at(n->input(1)).unwrapToInt();
            return a * b;
        },
        EvalOptions().validSchemas({"aten::mul.int(int a, int b) -> (int)"})
    }).evaluator({
        c10::Symbol::fromQualString("aten::sub"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            auto a = args.at(n->input(0)).unwrapToInt();
            auto b = args.at(n->input(1)).unwrapToInt();
            return a - b;
        },
        EvalOptions().validSchemas({"aten::sub.int(int a, int b) -> (int)"})
    }).evaluator({
        c10::Symbol::fromQualString("aten::__round_to_zero_floordiv"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            auto a = args.at(n->input(0)).unwrapToInt();
            auto b = args.at(n->input(1)).unwrapToInt();
            return a / b;
        },
        EvalOptions().validSchemas({"aten::__round_to_zero_floordiv(int a, int b) -> (int)"})
    }).evaluator({
        c10::Symbol::fromQualString("aten::slice"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            c10::List<c10::IValue> list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
            int64_t start = args.at(n->input(1)).unwrapToInt();
            int64_t end = args.at(n->input(2)).unwrapToInt();
            int64_t step = args.at(n->input(3)).unwrapToInt();

            const int64_t list_size = list.size();

            // clamp start and end to the bounds of the list
            const auto normalized_start =
                std::max((int64_t)0, normalizeIndex(start, list_size));
            const auto normalized_end =
                std::min(list_size, normalizeIndex(end, list_size));

            auto sliced_list = c10::impl::GenericList(list.elementType());
            if (normalized_end <= normalized_start) {
                // early exit if the slice is trivially empty
                return sliced_list;
            }

            sliced_list.reserve(normalized_end - normalized_start);

            for (auto i = normalized_start; i < normalized_end;) {
                sliced_list.push_back(list.get(i));
                i += step;
            }

            return sliced_list;
        },
        EvalOptions().validSchemas({"aten::slice.t(t[] l, int start, int end=9223372036854775807, int step=1) -> (t[])"})
    }).evaluator({
        c10::Symbol::fromQualString("aten::len"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            c10::List<c10::IValue> list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
            return static_cast<int64_t>(list.size());
        },
        EvalOptions().validSchemas({"aten::len.t(t[] a) -> (int)"})
    }).evaluator({
        c10::Symbol::fromQualString("aten::size"),
        [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
            LOG_WARNING("There may be undefined behavior using dynamic shape and aten::size");
            auto tensor_var = args.at(n->input(0));
            if (n->inputs().size() == 1) {
                if (tensor_var.isITensor()) {
                    auto tensor = tensor_var.ITensor();
                    return util::toVec(tensor->getDimensions());
                } else {
                    auto tensor = tensor_var.unwrapToTensor();
                    return tensor.sizes();
                }
            } else {
                auto dim = args.at(n->input(1)).unwrapToInt();
                if (tensor_var.isITensor()) {
                    auto tensor = tensor_var.ITensor();
                    return util::toVec(tensor->getDimensions())[dim];
                } else {
                    auto tensor = tensor_var.unwrapToTensor();
                    return tensor.sizes()[dim];
                }
            }
        },
        EvalOptions().validSchemas({
            "aten::size(Tensor self) -> (int[])",
            "aten::size.int(Tensor self, int dim) -> (int)"
        })
    });
}
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
