#include "ATen/core/List.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/stack.h"
#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/jit/ir/constants.h"
#include "torch/csrc/jit/ir/ir.h"
#include "torch/torch.h"

#include "core/conversion/evaluators/eval_macros.h"
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

DEFINE_GENERIC_TWO_INPUT_EVALUATOR(
    eq,
    "aten::eq",
    a == b,
    std::set<std::string>({
        "aten::eq.bool(bool a, bool b) -> (bool)",
        "aten::eq.int(int a, int b) -> (bool)",
        "aten::eq.float(float a, float b) -> (bool)",
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

DEFINE_TWO_INPUT_SIMPLE_EVALUATOR(and, "aten::__and__", a&& b, bool, {"aten::__and__(int a, int b) -> (bool)"});
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

auto aten_registrations TRTORCH_UNUSED =
    RegisterNodeEvaluators()
        .evaluator({c10::Symbol::fromQualString("aten::zeros"),
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
        .evaluator({c10::Symbol::fromQualString("aten::slice"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      c10::List<c10::IValue> list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
                      int64_t start = args.at(n->input(1)).unwrapToInt();
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
        .evaluator({c10::Symbol::fromQualString("aten::len"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      c10::List<c10::IValue> list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
                      return static_cast<int64_t>(list.size());
                    },
                    EvalOptions().validSchemas({"aten::len.t(t[] a) -> (int)"})})
        .evaluator({c10::Symbol::fromQualString("aten::size"),
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
                    EvalOptions().validSchemas(
                        {"aten::size(Tensor self) -> (int[])", "aten::size.int(Tensor self, int dim) -> (int)"})})
        .evaluator({c10::Symbol::fromQualString("aten::__getitem__"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
                      auto idx = args.at(n->input(1)).unwrapToInt();

                      const int64_t list_size = list.size();
                      const int64_t normalized_idx = normalizeIndex(idx, list_size);
                      TRTORCH_CHECK(
                          normalized_idx >= 0 || normalized_idx < list_size,
                          "List index out of range (aten::__getitem__)");
                      return list.get(normalized_idx);
                    },
                    EvalOptions().validSchemas({
                        "aten::__getitem__.t(t[](a) list, int idx) -> (t(*))",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::append"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto list = args.at(n->input(0)).IValue()->to<c10::List<c10::IValue>>();
                      auto el = args.at(n->input(1)).IValue();

                      list.push_back(std::move(*el));
                      return list;
                    },
                    EvalOptions().validSchemas({
                        "aten::append.t(t[](a!) self, t(c -> *) el) -> (t[](a!))",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::neg"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto el = args.at(n->input(0)).unwrapToInt();

                      return el * -1;
                    },
                    EvalOptions().validSchemas({
                        "aten::neg.int(int a) -> (int)",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::add"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      if (args.at(n->input(0)).IValue()->isInt()) {
                        auto a = args.at(n->input(0)).unwrapToInt();
                        auto b = args.at(n->input(1)).unwrapToInt();
                        return a + b;
                      } else if (args.at(n->input(0)).IValue()->isDouble()) {
                        auto a = args.at(n->input(0)).unwrapToDouble();
                        auto b = args.at(n->input(1)).unwrapToDouble();
                        return a + b;
                      } else {
                        TRTORCH_THROW_ERROR(
                            "Unimplemented data type for aten::add evaluator: "
                            << args.at(n->input(0)).IValue()->type()->str());
                        return {};
                      }
                    },
                    EvalOptions().validSchemas(
                        {"aten::add.int(int a, int b) -> (int)", "aten::add.float(float a, float b) -> (float)"})})
        .evaluator({c10::Symbol::fromQualString("aten::add_"),
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
                        TRTORCH_THROW_ERROR(
                            "Unimplemented data type for aten::add_ evaluator: "
                            << args.at(n->input(0)).IValue()->type()->str());
                        return {};
                      }
                    },
                    EvalOptions().validSchemas({"aten::add_.t(t[](a!) self, t[] b) -> (t[])"})})
        .evaluator({c10::Symbol::fromQualString("aten::mul"),
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
                        TRTORCH_THROW_ERROR(
                            "Unimplemented data type for aten::mul evaluator: "
                            << args.at(n->input(0)).IValue()->type()->str());
                        return {};
                      }
                    },
                    EvalOptions().validSchemas(
                        {"aten::mul.int(int a, int b) -> (int)", "aten::mul.float(float a, float b) -> (float)"})})
        .evaluator({c10::Symbol::fromQualString("aten::sub"),
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
                        TRTORCH_THROW_ERROR(
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
                 TRTORCH_THROW_ERROR(
                     "Unimplemented data type for aten::Bool evaluator: "
                     << args.at(n->input(0)).IValue()->type()->str());
                 return {};
               }
             },
             EvalOptions().validSchemas({"aten::Bool.int(int a) -> (bool)", "aten::Bool.float(float b) -> (bool)"})})
        .evaluator({c10::Symbol::fromQualString("aten::Float"),
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
                        TRTORCH_THROW_ERROR(
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
        .evaluator({c10::Symbol::fromQualString("aten::__not__"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto el = args.at(n->input(0)).unwrapToBool();

                      return !el;
                    },
                    EvalOptions().validSchemas({
                        "aten::__not__(bool self) -> bool",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::__is__"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto self = args.at(n->input(0)).IValue();
                      auto obj = args.at(n->input(1)).IValue();

                      return self->isSameIdentity(*obj);
                    },
                    EvalOptions().validSchemas({
                        "aten::__is__(t1 self, t2 obj) -> bool",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::__isnot__"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto self = args.at(n->input(0)).IValue();
                      auto obj = args.at(n->input(1)).IValue();

                      return !self->isSameIdentity(*obj);
                    },
                    EvalOptions().validSchemas({
                        "aten::__isnot__(t1 self, t2 obj) -> bool",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::numel"),
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
        .evaluator({c10::Symbol::fromQualString("aten::t"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto tensor_var = args.at(n->input(0));
                      if (tensor_var.IValue()->isTensor()) {
                        auto tensor = tensor_var.unwrapToTensor();
                        return tensor.t();
                      } else {
                        TRTORCH_THROW_ERROR("Unimplemented data type for aten::t evaluator: ITensor");
                        return {};
                      }
                    },
                    EvalOptions().validSchemas({
                        "aten::t(Tensor self) -> Tensor",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::dim"),
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
        .evaluator({c10::Symbol::fromQualString("aten::div"),
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
                        TRTORCH_THROW_ERROR(
                            "Unimplemented data type for aten::div evaluator: "
                            << args.at(n->input(0)).IValue()->type()->str());
                        return {};
                      }
                    },
                    EvalOptions().validSchemas({
                        "aten::div.float(float a, float b) -> (float)",
                        "aten::div.int(int a, int b) -> (float)",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::floordiv"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      if (args.at(n->input(0)).IValue()->isInt()) {
                        auto a = args.at(n->input(0)).unwrapToInt();
                        auto b = args.at(n->input(1)).unwrapToInt();
                        return std::floor(a / b);
                      } else if (args.at(n->input(0)).IValue()->isDouble()) {
                        auto a = args.at(n->input(0)).unwrapToDouble();
                        auto b = args.at(n->input(1)).unwrapToDouble();
                        return std::floor(a / b);
                      } else {
                        TRTORCH_THROW_ERROR(
                            "Unimplemented data type for aten::floordiv evaluator: "
                            << args.at(n->input(0)).IValue()->type()->str());
                        return {};
                      }
                    },
                    EvalOptions().validSchemas({
                        "aten::floordiv.float(float a, float b) -> (int)",
                        "aten::floordiv.int(int a, int b) -> (int)",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::floor"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      if (args.at(n->input(0)).IValue()->isInt()) {
                        auto el = args.at(n->input(0)).unwrapToInt();
                        return static_cast<int64_t>(std::floor(el));
                      } else if (args.at(n->input(0)).IValue()->isDouble()) {
                        auto el = args.at(n->input(0)).unwrapToDouble();
                        return static_cast<int64_t>(std::floor(el));
                      } else {
                        TRTORCH_THROW_ERROR(
                            "Unimplemented data type for aten::floor evaluator: "
                            << args.at(n->input(0)).IValue()->type()->str());
                        return {};
                      }
                    },
                    EvalOptions().validSchemas({
                        "aten::floor.int(int a) -> (int)",
                        "aten::floor.float(float a) -> (int)",
                    })})
        .evaluator({c10::Symbol::fromQualString("aten::warn"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      auto warning = args.at(n->input(0)).IValue();
                      LOG_WARNING("Warning from TorchScript: " << *warning);
                      return {};
                    },
                    EvalOptions()})
        .evaluator({c10::Symbol::fromQualString("aten::arange"),
                    [](const torch::jit::Node* n, kwargs& args) -> c10::optional<torch::jit::IValue> {
                      int input_size = n->inputs().size();
                      int scalar_count = 0;
                      for (int i = 0; i < input_size; i++) {
                        if (args.at(n->input(i)).IValue()->isScalar()) {
                          scalar_count += 1;
                        }
                      }
                      if (scalar_count == 1) {
                        if (args.at(n->input(0)).IValue()->isInt()) {
                          int end_scalar = args.at(n->input(0)).unwrapToInt();
                          return torch::arange(end_scalar);
                        } else if (args.at(n->input(0)).IValue()->isDouble()) {
                          float end_scalar = args.at(n->input(0)).unwrapToScalar().to<float>();
                          return torch::arange(end_scalar);
                        }
                      } else if (scalar_count == 2) {
                        if (args.at(n->input(0)).IValue()->isDouble() || args.at(n->input(1)).IValue()->isDouble()) {
                          float start_scalar = args.at(n->input(0)).unwrapToScalar().to<float>();
                          float end_scalar = args.at(n->input(1)).unwrapToScalar().to<float>();
                          return torch::arange(start_scalar, end_scalar);
                        } else {
                          int start_scalar = args.at(n->input(0)).unwrapToInt();
                          int end_scalar = args.at(n->input(1)).unwrapToInt();
                          return torch::arange(start_scalar, end_scalar);
                        }
                      } else if (scalar_count == 3) {
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
                        TRTORCH_THROW_ERROR(
                            "Invalid input argument size for aten::arange, input argument size: " << input_size);
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
                    })});
} // namespace
} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch
