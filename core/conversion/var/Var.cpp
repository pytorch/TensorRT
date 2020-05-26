#include "core/util/prelude.h"
#include "core/conversion/var/Var.h"

namespace trtorch {
namespace core {
namespace conversion {

Var::Var() {
  ptr_.none = nullptr;
  type_ = Type::kNone;
}

Var::Var(const torch::jit::IValue* p)
  : type_(Type::kIValue) {
  ptr_.ivalue = p;
}

Var::Var(nvinfer1::ITensor* p)
  : type_(Type::kITensor) {
  ptr_.tensor = p;
}

Var::Var(const Var& a) {
  switch(a.type_) {
  case Type::kITensor:
    ptr_.tensor = a.ptr_.tensor;
    type_ = Type::kITensor;
    break;
  case Type::kIValue:
    ptr_.ivalue = a.ptr_.ivalue;
    type_ = Type::kIValue;
    break;
  case Type::kNone:
  default:
    ptr_.none = a.ptr_.none;
    type_ = Type::kNone;
  }
}

Var& Var::operator=(const Var& a) {
  switch(a.type_) {
  case Type::kITensor:
    ptr_.tensor = a.ptr_.tensor;
    type_ = Type::kITensor;
    break;
  case Type::kIValue:
    ptr_.ivalue = a.ptr_.ivalue;
    type_ = Type::kIValue;
    break;
  case Type::kNone:
  default:
    ptr_.none = a.ptr_.none;
    type_ = Type::kNone;
  }
  return (*this);
}

Var& Var::operator=(const torch::jit::IValue* in) {
  ptr_.ivalue = in;
  type_ = Type::kIValue;
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

std::string Var::type_name() const {
  switch(type_) {
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

const torch::jit::IValue* Var::IValue() const {
  TRTORCH_CHECK(isIValue(), "Requested IValue from Var, however Var type is " << type_name());
  if (type_ == Type::kIValue) {
    return ptr_.ivalue;
  } else {
    return nullptr;
  }
}

nvinfer1::ITensor* Var::ITensor() const {
  TRTORCH_CHECK(isITensor(), "Requested ITensor from Var, however Var type is " << type_name());
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
} // namespace trtorch
