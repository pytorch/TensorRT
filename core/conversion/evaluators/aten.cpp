#include <math.h>

#include "ATen/core/List.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/stack.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/jit/ir/constants.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/torch.h"

#include "core/conversion/evaluators/eval_macros.h"
#include "core/conversion/evaluators/eval_util.h"
#include "core/conversion/evaluators/evaluators.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace evaluators {
namespace {

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    eq,
    "aten::eq",
    a == b,
    std::set<std::string>({
        "aten::eq.bool(bool a, bool b) -> (bool)",
        "aten::eq.int(int a, int b) -> (bool)",
        "aten::eq.float(float a, float b) -> (bool)",
        "aten::eq.str(str a, str b) -> (bool)",
        "aten::eq.int_float(int a, float b) -> (bool)",
        "aten::eq.float_int(float a, int b) -> (bool)",
    }));

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    ne,
    "aten::ne",
    a != b,
    std::set<std::string>({
        "aten::ne.bool(bool a, bool b) -> (bool)",
        "aten::ne.int(int a, int b) -> (bool)",
        "aten::ne.float(float a, float b) -> (bool)",
        "aten::ne.int_float(int a, float b) -> (bool)",
        "aten::ne.float_int(float a, int b) -> (bool)",
    }));

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    lt,
    "aten::lt",
    a < b,
    std::set<std::string>({
        "aten::lt.bool(bool a, bool b) -> (bool)",
        "aten::lt.int(int a, int b) -> (bool)",
        "aten::lt.float(float a, float b) -> (bool)",
        "aten::lt.int_float(int a, float b) -> (bool)",
        "aten::lt.float_int(float a, int b) -> (bool)",
    }));

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    gt,
    "aten::gt",
    a > b,
    std::set<std::string>({
        "aten::gt.bool(bool a, bool b) -> (bool)",
        "aten::gt.int(int a, int b) -> (bool)",
        "aten::gt.float(float a, float b) -> (bool)",
        "aten::gt.int_float(int a, float b) -> (bool)",
        "aten::gt.float_int(float a, int b) -> (bool)",
    }));

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    le,
    "aten::le",
    a <= b,
    std::set<std::string>({
        "aten::le.bool(bool a, bool b) -> (bool)",
        "aten::le.int(int a, int b) -> (bool)",
        "aten::le.float(float a, float b) -> (bool)",
        "aten::le.int_float(int a, float b) -> (bool)",
        "aten::le.float_int(float a, int b) -> (bool)",
    }));

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    ge,
    "aten::ge",
    a >= b,
    std::set<std::string>({
        "aten::ge.bool(bool a, bool b) -> (bool)",
        "aten::ge.int(int a, int b) -> (bool)",
        "aten::ge.float(float a, float b) -> (bool)",
        "aten::ge.int_float(int a, float b) -> (bool)",
        "aten::ge.float_int(float a, int b) -> (bool)",
    }));

DEFINE_ARITHMATIC_TWO_INPUT_EVALUATOR(
    pow,
    "aten::pow",
    pow(a, b),
    std::set<std::string>({
        "aten::pow.int(int a, int b) -> (float)",
        "aten::pow.float(float a, float b) -> (float)",
        "aten::pow.int_float(int a, float b) -> (float)",
        "aten::pow.float_int(float a, int b) -> (float)",
    }));

DEFINE_TWO_INPUT_SIMPLE_EVALUATOR(
    and,
    "aten::__and__",
    a&& b,
    bool,
    std::set<std::string>({"aten::__and__(int a, int b) -> (bool)", "aten::__and__.bool(bool a, bool b) -> (bool)"}));
DEFINE_TWO_INPUT_SIMPLE_EVALUATOR(or, "aten::__or__", a || b, bool, {"aten::__or__(int a, int b) -> (bool)"});
DEFINE_TWO_INPUT_SIMPLE_EVALUATOR(
    xor,
    "aten::__xor__",
    a != b,
    bool,
    {
        "aten::__xor__(int a, int b) -> (bool)"});
DEFINE_TWO_INPUT_SIMPLE_EVALUATOR(
    int_div,
    "aten::__round_to_zero_floordiv",
    a / b,
    int64_t,
    {"aten::__round_to_zero_floordiv(int a, int b) -> (int)"});

auto aten_registrations TORCHTRT_UNUSED =
    RegisterNodeEvaluators()
        .evaluator(
            {c10::Symbol::fromQualString("aten::zeros"),
             // aten::zeros(int[] size, *, int? dtype=None, int? layout=None,
             // Device? device=None, bool? pin_memory=None) -> (Tensor)
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto options = torch::TensorOptions().layout(torch::kStrided).device(torch::kCUDA);

               // Input 1 here is the dtype
               if (!args.at(n->input(1)).isNone() && !args.at(n->input(1)).IValue()->isNone()) {
                 options = options.dtype(c10::ScalarType(args.at(n->input(1)).unwrapToInt()));
               }

               auto out_tensor = torch::zeros(args.at(n->input(0)).unwrapToIntList().vec(), options);
               return out_tensor;
             }})
        .evaluator(
            {c10::Symbol::fromQualString("aten::ones"),
             // aten::ones(int[] size, *, int? dtype=None, int? layout=None,
             // Device? device=None, bool? pin_memory=None) -> (Tensor)
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto options = torch::TensorOptions().layout(torch::kStrided).device(torch::kCUDA);

               // Input 1 here is the dtype
               if (!args.at(n->input(1)).isNone() && !args.at(n->input(1)).IValue()->isNone()) {
                 options = options.dtype(c10::ScalarType(args.at(n->input(1)).unwrapToInt()));
               }

               auto out_tensor = torch::ones(args.at(n->input(0)).unwrapToIntList().vec(), options);
               return out_tensor;
             }})
        .evaluator(
            {c10::Symbol::fromQualString("aten::full"),
             // aten::full(int[] size, Scalar fill_value, *, int? dtype=None, int? layout=None,
             // Device? device=None, bool? pin_memory=None) -> (Tensor)
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto options = torch::TensorOptions().layout(torch::kStrided).device(torch::kCUDA);

               // Input 2 here is the dtype
               if (!args.at(n->input(2)).isNone() && !args.at(n->input(2)).IValue()->isNone()) {
                 options = options.dtype(c10::ScalarType(args.at(n->input(2)).unwrapToInt()));
               }

               auto scalar_value = args.at(n->input(1)).unwrapToScalar().to<float>();
               auto out_tensor = torch::full(args.at(n->input(0)).unwrapToIntList().vec(), scalar_value, options);
               return out_tensor;
             }})
        .evaluator(
            {c10::Symbol::fromQualString("aten::slice"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               c10::List<c10::IValue> list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();

               int64_t start = 0;
               auto startIVal = args.at(n->input(1)).IValue();
               if (!startIVal->isNone()) {
                 start = args.at(n->input(1)).unwrapToInt();
               }
               int64_t end = args.at(n->input(2)).unwrapToInt();
               int64_t step = args.at(n->input(3)).unwrapToInt();

               const int64_t list_size = list.size();

               // clamp start and end to the bounds of the list
               const auto normalized_start = std::max((int64_t)0, normalizeIndex(start, list_size));
               const auto normalized_end = std::min(list_size, normalizeIndex(end, list_size));

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
             EvalOptions().validSchemas(
                 {"aten::slice.t(t[] l, int start, int end=9223372036854775807, int step=1) -> (t[])"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::len"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               c10::List<c10::IValue> list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
               return static_cast<int64_t>(list.size());
             },
             EvalOptions().validSchemas({"aten::len.t(t[] a) -> (int)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::size"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               LOG_WARNING("There may be undefined behavior using dynamic shape and aten::size");
               auto tensor_var = args.at(n->input(0));
               if (n->inputs().size() == 1) {
                 if (tensor_var.isITensor()) {
                   auto tensor = tensor_var.ITensor();
                   return util::toVec(tensor->getDimensions());
                 } else if (tensor_var.IValue()->isTensor()) {
                   auto tensor = tensor_var.unwrapToTensor();
                   return tensor.sizes();
                 } else if (tensor_var.IValue()->isCustomClass()) {
                   auto tensor = tensor_var.IValue()->toCustomClass<TensorContainer>()->tensor();
                   return util::toVec(tensor->getDimensions());
                 } else {
                   TORCHTRT_THROW_ERROR("IValue is not some class of Tensor. Found: " << tensor_var.IValue()->type());
                 }
               } else {
                 auto dim = args.at(n->input(1)).unwrapToInt();
                 if (tensor_var.isITensor()) {
                   auto tensor = tensor_var.ITensor();
                   auto dims = util::toVec(tensor->getDimensions());
                   auto nbDims = tensor->getDimensions().nbDims;
                   if (dim < 0) {
                     dim += nbDims;
                   }
                   return dims[dim];
                 } else if (tensor_var.IValue()->isTensor()) {
                   auto tensor = tensor_var.unwrapToTensor();
                   auto nbDims = tensor.sizes().size();
                   if (dim < 0) {
                     dim += nbDims;
                   }
                   return tensor.sizes()[dim];
                 } else if (tensor_var.IValue()->isCustomClass()) {
                   auto tensor = tensor_var.IValue()->toCustomClass<TensorContainer>()->tensor();
                   auto dims = util::toVec(tensor->getDimensions());
                   auto nbDims = tensor->getDimensions().nbDims;
                   if (dim < 0) {
                     dim += nbDims;
                   }
                   return dims[dim];
                 } else {
                   TORCHTRT_THROW_ERROR("IValue is not some class of Tensor. Found: " << tensor_var.IValue()->type());
                 }
               }
             },
             EvalOptions().validSchemas(
                 {"aten::size(Tensor self) -> (int[])", "aten::size.int(Tensor self, int dim) -> (int)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::__getitem__"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
               auto idx = args.at(n->input(1)).unwrapToInt();

               const int64_t list_size = list.size();
               const int64_t normalized_idx = normalizeIndex(idx, list_size);
               TORCHTRT_CHECK(
                   normalized_idx >= 0 || normalized_idx < list_size, "List index out of range (aten::__getitem__)");
               return list.get(normalized_idx);
             },
             EvalOptions().validSchemas({
                 "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::append"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();

               if (args.at(n->input(1)).isITensor()) {
                 auto tensor_holder = TensorContainer();
                 tensor_holder.hold_tensor(args.at(n->input(1)).ITensor());
                 auto el = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
                 list.push_back(std::move(el));
               } else {
                 auto el = args.at(n->input(1)).IValue();
                 list.push_back(std::move(*el));
               }

               return list;
             },
             EvalOptions().validSchemas({
                 "aten::append.t(t[](a!) self, t(c -> *) el) -> (t[](a!))",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::extend"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isList() && args.at(n->input(1)).IValue()->isList()) {
                 c10::IValue* self_ptr = args.at(n->input(0)).IValueMut();
                 auto self = self_ptr->to<c10::List<c10::IValue>>();
                 auto other = args.at(n->input(1)).IValue()->to<c10::List<c10::IValue>>();
                 const int64_t other_size = other.size();

                 // Modify value in place
                 for (int64_t i = 0; i < other_size; i++) {
                   self.push_back(other.get(i));
                 }

                 *self_ptr = c10::IValue(self);
                 return {};
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::extend.t evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str() << ", "
                     << args.at(n->input(1)).IValue()->type()->str());
               }
             },
             EvalOptions().validSchemas({
                 "aten::extend.t(t[](a!) self, t[] other) -> ()",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::neg"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto el = args.at(n->input(0)).unwrapToInt();

               return el * -1;
             },
             EvalOptions().validSchemas({
                 "aten::neg.int(int a) -> (int)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::add"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 auto b = args.at(n->input(1)).unwrapToInt();
                 return a + b;
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 auto b = args.at(n->input(1)).unwrapToDouble();
                 return a + b;
               } else if (args.at(n->input(0)).IValue()->isString()) {
                 auto a = args.at(n->input(0)).unwrapToString();
                 auto b = args.at(n->input(1)).unwrapToString();
                 return a + b;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::add evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas(
                 {"aten::add.int(int a, int b) -> (int)",
                  "aten::add.float(float a, float b) -> (float)",
                  "aten::add.str(str a, str b) -> (str)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::add_"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isList()) {
                 auto a = args.at(n->input(0)).IValue()->toListRef();
                 auto b = args.at(n->input(1)).IValue()->toListRef();

                 c10::ListTypePtr lt = n->output()->type()->expect<c10::ListType>();
                 c10::TypePtr elementType = lt->getElementType();

                 auto merged = c10::impl::GenericList(elementType);
                 merged.reserve(a.size() + b.size());

                 for (auto each : a) {
                   merged.emplace_back(each);
                 }

                 for (auto each : b) {
                   merged.emplace_back(each);
                 }

                 return merged;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::add_ evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({"aten::add_.t(t[](a!) self, t[] b) -> (t[])"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::mul"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 auto b = args.at(n->input(1)).unwrapToInt();
                 return a * b;
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 auto b = args.at(n->input(1)).unwrapToDouble();
                 return a * b;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::mul evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas(
                 {"aten::mul.int(int a, int b) -> (int)", "aten::mul.float(float a, float b) -> (float)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::sub"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 auto b = args.at(n->input(1)).unwrapToInt();
                 return a - b;
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 auto b = args.at(n->input(1)).unwrapToDouble();
                 return a - b;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::sub evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::sub.float(float a, float b) -> (float)",
                 "aten::sub.int(int a, int b) -> (int)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::Bool"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 return (bool)a;
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 return (bool)a;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::Bool evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({"aten::Bool.int(int a) -> (bool)", "aten::Bool.float(float b) -> (bool)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::Float"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 return (float)a;
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 return (float)a;
               } else if (args.at(n->input(0)).IValue()->isBool()) {
                 auto a = args.at(n->input(0)).unwrapToBool();
                 return (float)a;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::Float evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::Float.Scalar(Scalar a) -> float",
                 "aten::Float.int(int a) -> float",
                 "aten::Float.bool(bool a) -> float",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::Int"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 return (int)a;
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 return (int)a;
               } else if (args.at(n->input(0)).IValue()->isBool()) {
                 auto a = args.at(n->input(0)).unwrapToBool();
                 return (int)a;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::Int evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::Int.Scalar(Scalar a) -> int",
                 "aten::Int.int(int a) -> int",
                 "aten::Int.bool(bool a) -> int",
                 "aten::Int.float(float a) -> int",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::__not__"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto el = args.at(n->input(0)).unwrapToBool();

               return !el;
             },
             EvalOptions().validSchemas({
                 "aten::__not__(bool self) -> bool",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::__is__"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto self = args.at(n->input(0)).IValue();
               auto obj = args.at(n->input(1)).IValue();

               return self->is(*obj);
             },
             EvalOptions().validSchemas({
                 "aten::__is__(t1 self, t2 obj) -> bool",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::__isnot__"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto self = args.at(n->input(0)).IValue();
               auto obj = args.at(n->input(1)).IValue();

               return !self->is(*obj);
             },
             EvalOptions().validSchemas({
                 "aten::__isnot__(t1 self, t2 obj) -> bool",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::numel"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               LOG_WARNING("There may be undefined behavior using dynamic shape and aten::numel");
               auto tensor_var = args.at(n->input(0));
               if (tensor_var.isITensor()) {
                 auto tensor = tensor_var.ITensor();
                 return util::volume(tensor->getDimensions());
               } else {
                 auto tensor = tensor_var.unwrapToTensor();
                 return tensor.numel();
               }
             },
             EvalOptions().validSchemas({
                 "aten::numel(Tensor self) -> int",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::dim"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto tensor_var = args.at(n->input(0));
               if (tensor_var.isITensor()) {
                 auto tensor = tensor_var.ITensor();
                 return tensor->getDimensions().nbDims;
               } else {
                 auto tensor = tensor_var.unwrapToTensor();
                 return tensor.dim();
               }
             },
             EvalOptions().validSchemas({
                 "aten::dim(Tensor self) -> int",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::div"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 auto b = args.at(n->input(1)).unwrapToInt();
                 return static_cast<double>(a) / static_cast<double>(b);
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 auto b = args.at(n->input(1)).unwrapToDouble();
                 return a / b;
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::div evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::div.float(float a, float b) -> (float)",
                 "aten::div.int(int a, int b) -> (float)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::floordiv"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 auto b = args.at(n->input(1)).unwrapToInt();
                 return static_cast<int>(std::floor(a / b));
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 auto b = args.at(n->input(1)).unwrapToDouble();
                 return std::floor(a / b);
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::floordiv evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::floordiv.float(float a, float b) -> (int)",
                 "aten::floordiv.int(int a, int b) -> (int)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::floor"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto el = args.at(n->input(0)).unwrapToInt();
                 return static_cast<int64_t>(std::floor(el));
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto el = args.at(n->input(0)).unwrapToDouble();
                 return static_cast<int64_t>(std::floor(el));
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::floor evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::floor.int(int a) -> (int)",
                 "aten::floor.float(float a) -> (int)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::sqrt"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).IValue()->isInt()) {
                 auto a = args.at(n->input(0)).unwrapToInt();
                 return std::sqrt(static_cast<double>(a));
               } else if (args.at(n->input(0)).IValue()->isDouble()) {
                 auto a = args.at(n->input(0)).unwrapToDouble();
                 return std::sqrt(a);
               } else {
                 TORCHTRT_THROW_ERROR(
                     "Unimplemented data type for aten::sqrt evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({
                 "aten::sqrt.int(int a) -> (float)",
                 "aten::sqrt.float(float a) -> (float)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::warn"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto warning = args.at(n->input(0)).IValue();
               LOG_WARNING("Warning from TorchScript: " << *warning);
               return {};
             },
             EvalOptions()})
        .evaluator(
            {c10::Symbol::fromQualString("aten::is_floating_point"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto tensor_var = args.at(n->input(0));
               if (tensor_var.isITensor()) {
                 auto tensor = tensor_var.ITensor();
                 auto t = tensor->getType();
                 return (t == nvinfer1::DataType::kFLOAT || t == nvinfer1::DataType::kHALF);
               } else {
                 auto tensor = tensor_var.unwrapToTensor();
                 auto t = tensor.scalar_type();
                 return at::isFloatingType(t);
               }
             },
             EvalOptions().validSchemas({
                 "aten::is_floating_point(Tensor self) -> (bool)",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::tensor"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto data = args.at(n->input(0)).IValue();
               auto dtype = args.at(n->input(1)).IValue();
               auto device = args.at(n->input(2)).IValue();
               auto tensor = createTensorFromList(*data, *dtype, *device);
               return tensor;
             },
             EvalOptions().validSchemas(
                 {"aten::tensor(t[] data, *, int? dtype=None, Device? device=None, bool requires_grad=False) -> (Tensor)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::arange"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto schema = n->maybeSchema();
               TORCHTRT_CHECK(schema, "Unable to get schema for node: " << *n);
               auto name = schema->operator_name();

               if (c10::toString(name) == "aten::arange") {
                 if (args.at(n->input(0)).IValue()->isInt()) {
                   int end_scalar = args.at(n->input(0)).unwrapToInt();
                   return torch::arange(end_scalar);
                 } else if (args.at(n->input(0)).IValue()->isDouble()) {
                   float end_scalar = args.at(n->input(0)).unwrapToScalar().to<float>();
                   return torch::arange(end_scalar);
                 }
               } else if (c10::toString(name) == "aten::arange.start") {
                 if (args.at(n->input(0)).IValue()->isDouble() || args.at(n->input(1)).IValue()->isDouble()) {
                   float start_scalar = args.at(n->input(0)).unwrapToScalar().to<float>();
                   float end_scalar = args.at(n->input(1)).unwrapToScalar().to<float>();
                   return torch::arange(start_scalar, end_scalar);
                 } else {
                   int start_scalar = args.at(n->input(0)).unwrapToInt();
                   int end_scalar = args.at(n->input(1)).unwrapToInt();
                   return torch::arange(start_scalar, end_scalar);
                 }
               } else if (c10::toString(name) == "aten::arange.start_step") {
                 if (args.at(n->input(0)).IValue()->isDouble() || args.at(n->input(1)).IValue()->isDouble() ||
                     args.at(n->input(2)).IValue()->isDouble()) {
                   float start_scalar = args.at(n->input(0)).unwrapToScalar().to<float>();
                   float end_scalar = args.at(n->input(1)).unwrapToScalar().to<float>();
                   float step_scalar = args.at(n->input(2)).unwrapToScalar().to<float>();
                   return torch::arange(start_scalar, end_scalar, step_scalar);
                 } else {
                   int start_scalar = args.at(n->input(0)).unwrapToInt();
                   int end_scalar = args.at(n->input(1)).unwrapToInt();
                   int step_scalar = args.at(n->input(2)).unwrapToInt();
                   return torch::arange(start_scalar, end_scalar, step_scalar);
                 }
               } else {
                 TORCHTRT_THROW_ERROR("Unsupported aten::arange variant: " << name);
               }
               return {};
             },
             EvalOptions().validSchemas({
                 R"SIG(aten::arange(Scalar end, *, int? dtype=None, int? layout=None,
                            Device? device=None, bool? pin_memory=None) -> (Tensor))SIG",
                 R"SIG(aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None,
                            Layout? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor))SIG",
                 R"SIG(aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None,
                        Layout? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor))SIG",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::clone"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(0)).isITensor()) {
                 auto source_tensor = args.at(n->input(0)).ITensor();
                 auto tensor_holder = TensorContainer();
                 tensor_holder.hold_tensor(source_tensor);
                 auto clone_tensor = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
                 return std::move(clone_tensor);
               } else {
                 auto source_tensor = args.at(n->input(0)).unwrapToTensor();
                 auto clone_tensor = source_tensor.clone();
                 return clone_tensor;
               }
             },
             EvalOptions().validSchemas({
                 R"SIG(aten::clone(Tensor self, *, int? memory_format=None) -> (Tensor))SIG",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::copy_"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               if (args.at(n->input(1)).isITensor()) {
                 auto source_tensor = args.at(n->input(1)).ITensor();
                 auto tensor_holder = TensorContainer();
                 tensor_holder.hold_tensor(source_tensor);
                 auto clone_tensor = c10::IValue(std::move(c10::make_intrusive<TensorContainer>(tensor_holder)));
                 return std::move(clone_tensor);
               } else {
                 auto source_tensor = args.at(n->input(1)).unwrapToTensor();
                 auto self_tensor = args.at(n->input(0)).unwrapToTensor();
                 self_tensor.copy_(source_tensor);
                 return self_tensor;
               }
             },
             EvalOptions().validSchemas({
                 R"SIG(aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> (Tensor(a!)))SIG",
             })})
        .evaluator(
            {c10::Symbol::fromQualString("aten::format"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               int64_t input_num = n->inputs().size();
               std::vector<torch::jit::IValue> stack;
               for (auto v : n->inputs()) {
                 stack.push_back(*args.at(v).IValue());
               }
               stack.push_back(input_num);
               auto& ops = torch::jit::getAllOperatorsFor(c10::Symbol::fromQualString("aten::format"));
               auto& aten_format = ops.front();
               aten_format->getOperation()(stack);
               std::string output;
               torch::jit::pop(stack, output);
               return output;
             },
             EvalOptions().validSchemas({"aten::format(str self, ...) -> (str)"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::__range_length"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto lo = args.at(n->input(0)).unwrapToInt();
               auto hi = args.at(n->input(1)).unwrapToInt();
               auto step = args.at(n->input(2)).unwrapToInt();

               if (step == 0) {
                 TORCHTRT_THROW_ERROR("aten::__range_length() arg 3 must not be zero");
               }
               if (step > 0 && lo < hi) {
                 return 1 + (hi - 1 - lo) / step;
               } else if (step < 0 && lo > hi) {
                 return 1 + (lo - 1 - hi) / (0 - step);
               } else {
                 return 0;
               }
             },
             EvalOptions().validSchemas({"aten::__range_length(int lo, int hi, int step) -> int"})})
        .evaluator(
            {c10::Symbol::fromQualString("aten::__derive_index"),
             [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
               auto idx = args.at(n->input(0)).unwrapToInt();
               auto start = args.at(n->input(1)).unwrapToInt();
               auto step = args.at(n->input(2)).unwrapToInt();
               return start + idx * step;
             },
             EvalOptions().validSchemas({"aten::__derive_index(int idx, int start, int step) -> int"})});

} // namespace
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt