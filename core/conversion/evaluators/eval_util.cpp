#include <ATen/ATen.h>
#include "ATen/InitialTensorOptions.h"
#include "ATen/core/List.h"
#include "ATen/core/functional.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/jit_type.h"
#include "c10/util/irange.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace evaluators {

int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

// TODO: Switch back to PyTorch canonical implimentation
c10::optional<torch::jit::IValue> toIValue(const torch::jit::Value* v) {
  if (v->node()->kind() != torch::jit::prim::Constant || v->type()->cast<c10::FunctionType>()) {
    return c10::nullopt;
  }
  const torch::jit::Node* node = v->node();
  const c10::TypePtr& type = v->type();
  if (type->isSubtypeOf(c10::TensorType::get())) {
    return node->t(c10::attr::value);
  } else if (type->isSubtypeOf(c10::BoolType::get())) {
    return (bool)node->i(c10::attr::value);
  } else if (
      type->isSubtypeOf(c10::NumberType::get()) && node->kindOf(c10::attr::value) == torch::jit::AttributeKind::i) {
    return node->i(c10::attr::value);
  } else if (
      type->isSubtypeOf(c10::NumberType::get()) && node->kindOf(c10::attr::value) == torch::jit::AttributeKind::f) {
    return node->f(c10::attr::value);
  } else if (type->isSubtypeOf(c10::ListType::ofInts())) {
    try {
      const auto& is = node->is(c10::attr::value);
      return is;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->isSubtypeOf(c10::ListType::ofFloats())) {
    try {
      const auto& fs = node->fs(c10::attr::value);
      return fs;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->isSubtypeOf(c10::ListType::ofBools())) {
    const auto bs = c10::fmap<bool>(node->is(c10::attr::value));
    return bs;
  } else if (type->isSubtypeOf(c10::ListType::ofTensors())) {
    try {
      const auto& ts = node->ts(c10::attr::value);
      return ts;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->isSubtypeOf(c10::ListType::ofStrings())) {
    try {
      const auto& ss = node->ss(c10::attr::value);
      auto vals = c10::impl::GenericList(c10::StringType::get());
      for (const auto& str : ss) {
        vals.push_back(str);
      }
      return vals;
    } catch (const std::exception& ex) {
      const auto& ival = node->ival(c10::attr::value);
      return ival;
    }
  } else if (type->cast<c10::ListType>() && node->kindOf(c10::attr::value) == torch::jit::AttributeKind::ival) {
    const auto& list = node->ival(c10::attr::value);
    TORCHTRT_ASSERT(list.isList(), "Is not a list");
    return list;
  } else if (type->cast<c10::DictType>() && node->kindOf(c10::attr::value) == torch::jit::AttributeKind::ival) {
    const auto& dict = node->ival(c10::attr::value);
    TORCHTRT_ASSERT(dict.isGenericDict(), "Is not a dict");
    return dict;
  } else if (type->cast<c10::TupleType>() && node->kindOf(c10::attr::value) == torch::jit::AttributeKind::ival) {
    const auto& tup = node->ival(c10::attr::value);
    TORCHTRT_ASSERT(tup.isTuple(), "Is not a tuple");
    return tup;
  } else if (type == c10::StringType::get()) {
    const auto& s = node->s(c10::attr::value);
    return s;
  } else if (type == c10::DeviceObjType::get()) {
    auto d = c10::Device(node->s(c10::attr::value));
    return d;
  } else if (node->mustBeNone()) {
    return torch::jit::IValue();
  } else {
    std::stringstream ss;
    ss << "constant literal not supported for: " << type->str();
    throw std::runtime_error(ss.str());
  }
}

void checkListInputType(const c10::TypePtr& elem_type, bool empty_list) {
  if (!elem_type->isSubtypeOf(c10::NumberType::get()) && elem_type != c10::BoolType::get()) {
    std::stringstream error;
    error << "Input must be of ints, floats, or bools, "
          << "got " << elem_type->repr_str();
    // special case empty list torch.tensor([])
    if (elem_type->isSubtypeOf(c10::TensorType::get())) {
      if (empty_list) {
        error << "\nEmpty lists default to List[Tensor]. Add a variable "
                 "annotation to the assignment to create an empty list "
                 "of another type (torch.jit.annotate(List[T, []]) where T "
                 "is the type of elements in the list for Python 2)";
      }
    }
    TORCHTRT_THROW_ERROR(error.str());
  }
}

void checkSequenceSize(int64_t n, int64_t dim, int64_t seq_size) {
  if (seq_size != n) {
    TORCHTRT_THROW_ERROR("Expected sequence of length " << n << " at dim " << dim << " (got " << seq_size << ")");
  }
}

// TODO: Conditionally enable truncation based on user setting
at::Tensor scalar_to_tensor(const at::Scalar& s, const at::Device device = at::kCPU) {
  // This function is basically same with the one in
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/ScalarOps.h, what different here is that Int and Float
  // won't be upgraded to kDouble or kLong since we don't support these 2 types in conversion
  if (device == at::kCPU) {
    if (s.isFloatingPoint()) {
      LOG_WARNING("Unable to process input type of at::kDouble, truncate type to at::kFloat in scalar_to_tensor_util ");
      return at::detail::scalar_tensor_static(s, at::kFloat, at::kCPU);
    } else if (s.isComplex()) {
      return at::detail::scalar_tensor_static(s, at::kComplexDouble, at::kCPU);
    } else if (s.isBoolean()) {
      return at::detail::scalar_tensor_static(s, at::kBool, at::kCPU);
    } else {
      AT_ASSERT(s.isIntegral(false));
      LOG_WARNING("Unable to process input type of at::kLong, truncate type to at::kInt in scalar_to_tensor_util ");
      return at::detail::scalar_tensor_static(s, at::kInt, at::kCPU);
    }
  }
  if (s.isFloatingPoint()) {
    LOG_WARNING("Unable to process input type of at::kDouble, truncate type to at::kFloat in scalar_to_tensor_util ");
    return at::scalar_tensor(s, at::device(device).dtype(at::kFloat));
  } else if (s.isBoolean()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kBool));
  } else if (s.isComplex()) {
    return at::scalar_tensor(s, at::device(device).dtype(at::kComplexDouble));
  } else {
    AT_ASSERT(s.isIntegral(false));
    LOG_WARNING("Unable to process input type of at::kLong, truncate type to at::kInt in scalar_to_tensor_util ");
    return at::scalar_tensor(s, at::device(device).dtype(at::kInt));
  }
}

template <typename DTYPE>
void storeLastDimension(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<torch::jit::IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(DTYPE*)data = obj[i].to<DTYPE>();
    data += strides[dim] * elementSize;
  }
}

void storeLastDimensionFloat(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<torch::jit::IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(float*)data = static_cast<float>(obj[i].to<double>());
    data += strides[dim] * elementSize;
  }
}

void storeLastDimensionHalf(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    at::ArrayRef<torch::jit::IValue> obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(at::Half*)data = at::convert<at::Half, double>(obj[i].to<double>());
    data += strides[dim] * elementSize;
  }
}

void recursiveStore(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int tenElementSize,
    const torch::jit::IValue& obj) {
  auto ndim = sizes.size();
  auto n = sizes[dim];
  auto seq = obj.toListRef();
  checkSequenceSize(n, dim, seq.size());
  if (dim + 1 < static_cast<long>(ndim)) {
    for (const auto i : c10::irange(n)) {
      recursiveStore(data, sizes, strides, dim + 1, tenElementSize, seq[i]);
      data += strides[dim] * tenElementSize;
    }
  } else {
    if (obj.isIntList()) {
      storeLastDimension<int64_t>(data, sizes, strides, dim, tenElementSize, seq);
    } else if (obj.isBoolList()) {
      storeLastDimension<bool>(data, sizes, strides, dim, tenElementSize, seq);
    } else if (obj.isDoubleList()) {
      if (tenElementSize == static_cast<int>(c10::elementSize(at::ScalarType::Double))) {
        storeLastDimension<double>(data, sizes, strides, dim, tenElementSize, seq);
      } else if (tenElementSize == static_cast<int>(c10::elementSize(at::ScalarType::Float))) {
        storeLastDimensionFloat(data, sizes, strides, dim, tenElementSize, seq);
      } else if (tenElementSize == static_cast<int>(c10::elementSize(at::ScalarType::Half))) {
        storeLastDimensionHalf(data, sizes, strides, dim, tenElementSize, seq);
      } else {
        TORCHTRT_THROW_ERROR("Found unsupported data type in arguments for aten::tensor");
      }
    } else {
      TORCHTRT_THROW_ERROR("Found unsupported data type in arguments for aten::tensor");
    }
  }
}

at::Tensor castTensorTo(at::Tensor self, const torch::jit::IValue& dtype, const torch::jit::IValue& device) {
  at::ScalarType scalar_type = dtype.isNone() ? self.scalar_type() : dtype.toScalarType();
  c10::Device dev = device.isNone() ? self.device() : device.toDevice();
  if (scalar_type != self.scalar_type() || dev != self.device()) {
    self = self.to(dev, scalar_type);
  }
  return self;
}

std::vector<int64_t> compute_sizes(const torch::jit::IValue& seq) {
  std::vector<int64_t> sizes;
  auto seq_recur = seq.toList();
  while (true) {
    sizes.push_back(seq_recur.size());
    if (seq_recur.size() == 0 || !seq_recur.get(0).isList()) {
      break;
    }
    seq_recur = seq_recur.get(0).toList();
  }
  return sizes;
}

at::Tensor createTensorFromList(
    const torch::jit::IValue& data,
    const torch::jit::IValue& dtype,
    const torch::jit::IValue& device) {
  auto elem_type = data.type();
  /// Recurse down nested lists to find base type
  while (auto list_type = elem_type->cast<c10::ListType>()) {
    elem_type = list_type->getElementType();
  }
  /// Gets shape of tensor to be created
  auto sizes = compute_sizes(data);
  checkListInputType(elem_type, sizes.size() == 1 && sizes[0] == 0);
  at::ScalarType initial_scalar_type = c10::scalarTypeFromJitType(*elem_type);
  if (initial_scalar_type == at::ScalarType::Double) {
    initial_scalar_type = at::typeMetaToScalarType(c10::get_default_dtype());
  }

  auto tensor = at::empty(sizes, at::initialTensorOptions().dtype(initial_scalar_type));

  if (tensor.numel() != 0) {
    recursiveStore((char*)tensor.data_ptr(), sizes, tensor.strides(), 0, tensor.element_size(), data);
  }

  tensor = castTensorTo(tensor, dtype, device);
  auto default_type = at::typeMetaToScalarType(at::get_default_dtype());

  if (dtype.isNone() && tensor.scalar_type() != default_type && tensor.numel() == 0) {
    LOG_WARNING(
        "Creating a tensor from an empty "
        << elem_type->repr_str() << "list will create a tensor of default floating point type  (currently "
        << default_type << ") in python but a tensor of type " << elem_type->repr_str() << " in torchscript.\n"
        << "Pass in a dtype argument to ensure consistent behavior");
  }

  return tensor;
}

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
