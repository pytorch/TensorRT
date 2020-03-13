#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {

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
    if (type_ == Type::kIValue) {
        return ptr_.ivalue;
    } else {
        return nullptr;
    }
}

nvinfer1::ITensor* Arg::ITensor() const {
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

at::Tensor Arg::unwrapToTensor(at::Tensor default_val) {
    return this->unwrapTo<at::Tensor>(default_val);
}

at::Tensor Arg::unwrapToTensor() {
    return this->unwrapTo<at::Tensor>();
}

int64_t Arg::unwrapToInt(int64_t default_val) {
    return this->unwrapTo<int64_t>(default_val);
}

int64_t Arg::unwrapToInt() {
    return this->unwrapTo<int64_t>();
}

double Arg::unwrapToDouble(double default_val) {
    return this->unwrapTo<double>(default_val);
}

double Arg::unwrapToDouble() {
    return this->unwrapTo<double>();
} 

bool Arg::unwrapToBool(bool default_val) {
    return this->unwrapTo<bool>(default_val);
}

bool Arg::unwrapToBool() {
    return this->unwrapTo<bool>();
}

c10::Scalar Arg::unwrapToScalar(c10::Scalar default_val) {
    return this->unwrapTo<c10::Scalar>(default_val);
}

c10::Scalar Arg::unwrapToScalar() {
    return this->unwrapTo<c10::Scalar>();
}

c10::List<int64_t> Arg::unwrapToIntList(c10::List<int64_t> default_val) {
    return this->unwrapTo<c10::List<int64_t>>(default_val);
}

c10::List<int64_t> Arg::unwrapToIntList() {
    return this->unwrapTo<c10::List<int64_t>>();
}

c10::List<double> Arg::unwrapToDoubleList(c10::List<double> default_val) {
    return this->unwrapTo<c10::List<double>>(default_val);
}

c10::List<double> Arg::unwrapToDoubleList() {
    return this->unwrapTo<c10::List<double>>();
}

c10::List<bool> Arg::unwrapToBoolList(c10::List<bool> default_val) {
    return this->unwrapTo<c10::List<bool>>(default_val);
}

c10::List<bool> Arg::unwrapToBoolList() {
    return this->unwrapTo<c10::List<bool>>();
}

template<typename T>
T Arg::unwrapTo(T default_val) {
    if (isIValue()) {
        // TODO: implement Tag Checking 
        return ptr_.ivalue->to<T>();
    }
    LOG_DEBUG("In arg unwrapping, returning default value provided");
    return default_val;
}


template<typename T>
T Arg::unwrapTo() {
    if (isIValue()) {
        //TODO: Implement Tag checking
        return ptr_.ivalue->to<T>();
        //TODO: Exception
        //LOG_INTERNAL_ERROR("Requested unwrapping of arg IValue assuming it was " << typeid(T).name() << " however type is " << ptr_.ivalue->type());

    }
    TRTORCH_THROW_ERROR("Requested unwrapping of arg assuming it was an IValue, however arg type is " << type_name());
    return T();
}



} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
