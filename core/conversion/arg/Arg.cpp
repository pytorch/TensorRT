#include "core/util/prelude.h"
#include "core/conversion/arg/Arg.h"

namespace trtorch {
namespace core {
namespace conversion {

Arg::Arg() {
  ptr_.none = nullptr;
  type_ = Type::kNone;
}

Arg::Arg(const torch::jit::IValue* p)
  : type_(Type::kIValue) {
  ptr_.ivalue = p;
}

Arg::Arg(nvinfer1::ITensor* p)
  : type_(Type::kITensor) {
  ptr_.tensor = p;
}

Arg::Arg(const Arg& a) {
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

Arg& Arg::operator=(const Arg& a) {
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

Arg& Arg::operator=(const torch::jit::IValue* in) {
  ptr_.ivalue = in;
  type_ = Type::kIValue;
  return (*this);
}

Arg& Arg::operator=(nvinfer1::ITensor* in) {
  ptr_.tensor = in;
  type_ = Type::kITensor;
  return (*this);
}

Arg::Type Arg::type() const {
  return type_;
}

std::string Arg::type_name() const {
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

const torch::jit::IValue* Arg::IValue() const {
  TRTORCH_CHECK(isIValue(), "Requested IValue from Arg, however arg type is " << type_name());
  if (type_ == Type::kIValue) {
    return ptr_.ivalue;
  } else {
    return nullptr;
  }
}

nvinfer1::ITensor* Arg::ITensor() const {
  TRTORCH_CHECK(isITensor(), "Requested ITensor from Arg, however arg type is " << type_name());
  if (type_ == Type::kITensor) {
    return ptr_.tensor;
  } else {
    return nullptr;
  }
}

bool Arg::isITensor() const {
  if (type_ == Type::kITensor) {
    return true;
  } else {
    return false;
  }
}

bool Arg::isIValue() const {
  if (type_ == Type::kIValue) {
    return true;
  } else {
    return false;
  }
}

bool Arg::isNone() const {
  if (type_ == Type::kNone) {
    return true;
  } else {
    return false;
  }
}

} // namespace conversion
} // namespace core
} // namespace trtorch
