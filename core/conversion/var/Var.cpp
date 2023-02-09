#include <sstream>

#include "core/conversion/converters/converter_util.h"
#include "core/conversion/var/Var.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {

Var::Var() {
  ptr_.none = nullptr;
  type_ = Type::kNone;
}

Var::Var(torch::jit::IValue* p) : type_(Type::kIValue) {
  ptr_.ivalue = p;
}

Var::Var(nvinfer1::ITensor* p) : type_(Type::kITensor) {
  ptr_.tensor = p;
}

Var::IValueType Var::determineIValueType(torch::jit::IValue* p) {
  if (p->isInt()) {
    return IValueType::kInt;
  } else if (p->isDouble()) {
    return IValueType::kDouble;
  } else if (p->isBool()) {
    return IValueType::kBool;
  } else if (p->isTensor()) {
    return IValueType::kTensor;
  } else if (p->isIntList()) {
    return IValueType::kIntList;
  } else if (p->isDoubleList()) {
    return IValueType::kDoubleList;
  } else if (p->isBoolList()) {
    return IValueType::kBoolList;
  } else if (p->isTensorList()) {
    return IValueType::kTensorList;
  } else if (p->isList()) {
    return IValueType::kITensorList;
  }
}

Var::Var(const Var& a) {
  switch (a.type_) {
    case Type::kITensor:
      ptr_.tensor = a.ptr_.tensor;
      type_ = Type::kITensor;
      break;
    case Type::kIValue:
      ptr_.ivalue = a.ptr_.ivalue;
      type_ = Type::kIValue;
      ivalue_type_ = determineIValueType(ptr_.ivalue);
      break;
    case Type::kNone:
    default:
      ptr_.none = a.ptr_.none;
      type_ = Type::kNone;
  }
}

Var& Var::operator=(const Var& a) {
  switch (a.type_) {
    case Type::kITensor:
      ptr_.tensor = a.ptr_.tensor;
      type_ = Type::kITensor;
      break;
    case Type::kIValue:
      ptr_.ivalue = a.ptr_.ivalue;
      type_ = Type::kIValue;
      ivalue_type_ = determineIValueType(ptr_.ivalue);
      break;
    case Type::kNone:
    default:
      ptr_.none = a.ptr_.none;
      type_ = Type::kNone;
  }
  return (*this);
}

Var& Var::operator=(torch::jit::IValue* in) {
  ptr_.ivalue = in;
  type_ = Type::kIValue;
  ivalue_type_ = determineIValueType(ptr_.ivalue);
  return (*this);
}

Var& Var::operator=(nvinfer1::ITensor* in) {
  ptr_.tensor = in;
  type_ = Type::kITensor;
  return (*this);
}

Var::Type Var::type() const {
  return type_;
}

Var::IValueType Var::ivalue_type() const {
  return ivalue_type_;
}

std::string Var::type_name() const {
  switch (type_) {
    case Type::kITensor:
      return "nvinfer1::ITensor";
      break;
    case Type::kIValue:
      return "c10::IValue";
      break;
    case Type::kNone:
    default:
      return "None";
  }
}

nvinfer1::ITensor* Var::ITensorOrFreeze(ConversionCtx* ctx) {
  if (isIValue()) {
    LOG_DEBUG(ctx->logger, "Found IValue containing object of type " << *(ptr_.ivalue->type()));
  }

  TORCHTRT_CHECK(
      isITensor() || (isIValue() && (ptr_.ivalue->isTensor() || ptr_.ivalue->isCustomClass())),
      "Requested either IValue containing a Tensor, or ITensor, however Var type is " << type_name());

  nvinfer1::ITensor* out;

  if (isIValue()) {
    if (ptr_.ivalue->isTensor()) {
      auto tensor = ptr_.ivalue->toTensor();
      out = converters::tensor_to_const(ctx, tensor);
    } else {
      // Split converter generates c10::IValue which hold TensorContainer.
      auto output_container = ptr_.ivalue->toCustomClass<TensorContainer>();
      out = output_container.get()->tensor();
    }
  } else {
    out = ptr_.tensor;
  }

  LOG_DEBUG("ITensor name: " << out->getName());
  LOG_DEBUG("ITensor shape: " << out->getDimensions());
  LOG_DEBUG("ITensor type: " << out->getType());
  return out;
}

const torch::jit::IValue* Var::IValue() const {
  return IValueMut();
}

torch::jit::IValue* Var::IValueMut() const {
  TORCHTRT_CHECK(isIValue(), "Requested IValue from Var, however Var type is " << type_name());
  if (type_ == Type::kIValue) {
    return ptr_.ivalue;
  } else {
    return nullptr;
  }
}

nvinfer1::ITensor* Var::ITensor() const {
  TORCHTRT_CHECK(isITensor(), "Requested ITensor from Var, however Var type is " << type_name());
  if (type_ == Type::kITensor) {
    return ptr_.tensor;
  } else {
    return nullptr;
  }
}

bool Var::isITensor() const {
  if (type_ == Type::kITensor) {
    return true;
  } else {
    return false;
  }
}

bool Var::isITensorList() const {
  if (ivalue_type_ == IValueType::kITensorList) {
    return true;
  } else {
    return false;
  }
}

bool Var::isIntList() const {
  if (ivalue_type_ == IValueType::kIntList) {
    return true;
  } else {
    return false;
  }
}

bool Var::isDoubleList() const {
  if (ivalue_type_ == IValueType::kDoubleList) {
    return true;
  } else {
    return false;
  }
}

bool Var::isTensorList() const {
  if (ivalue_type_ == IValueType::kTensorList) {
    return true;
  } else {
    return false;
  }
}

bool Var::isBoolList() const {
  if (ivalue_type_ == IValueType::kBoolList) {
    return true;
  } else {
    return false;
  }
}

std::vector<nvinfer1::ITensor*> Var::unwrapToITensorList() {
  TORCHTRT_CHECK(
      isIValue(), "Requested unwrapping of arg assuming it was an IValue, however arg type is " << type_name());
  TORCHTRT_CHECK(
      isITensorList(),
      "Expected IValue to be an ITensorList, however the type is "
          << static_cast<std::underlying_type<IValueType>::type>(ivalue_type_));
  auto ivalue_list = ptr_.ivalue->toList();
  std::vector<nvinfer1::ITensor*> outputs;
  for (int i = 0; i < ivalue_list.size(); i++) {
    auto element = ivalue_list.get(i).toCustomClass<TensorContainer>()->tensor();
    outputs.push_back(std::move(element));
  }
  return outputs;
}

bool Var::isIValue() const {
  if (type_ == Type::kIValue) {
    return true;
  } else {
    return false;
  }
}

bool Var::isNone() const {
  if (type_ == Type::kNone) {
    return true;
  } else {
    return false;
  }
}

} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
