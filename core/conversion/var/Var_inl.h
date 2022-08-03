#pragma once

namespace torch_tensorrt {
namespace core {
namespace conversion {

#define DEFINE_UNWRAP_TO(ival_type, method_variant)                                                                \
  template <>                                                                                                      \
  inline ival_type Var::unwrapTo<ival_type>() {                                                                    \
    TORCHTRT_CHECK(                                                                                                \
        isIValue(), "Requested unwrapping of arg assuming it was an IValue, however arg type is " << type_name()); \
    auto ivalue = ptr_.ivalue;                                                                                     \
    TORCHTRT_CHECK(                                                                                                \
        ivalue->is##method_variant(),                                                                              \
        "Requested unwrapping of arg IValue assuming it was " << typeid(ival_type).name() << " however type is "   \
                                                              << *(ptr_.ivalue->type()));                          \
    return ptr_.ivalue->to<ival_type>();                                                                           \
  }                                                                                                                \
  template <>                                                                                                      \
  inline ival_type Var::unwrapTo(ival_type default_val) {                                                          \
    try {                                                                                                          \
      return this->unwrapTo<ival_type>();                                                                          \
    } catch (torch_tensorrt::Error & e) {                                                                          \
      LOG_DEBUG("In arg unwrapping, returning default value provided (" << e.what() << ")");                       \
      return default_val;                                                                                          \
    }                                                                                                              \
  }                                                                                                                \
                                                                                                                   \
  inline ival_type Var::unwrapTo##method_variant(ival_type default_val) {                                          \
    return this->unwrapTo<ival_type>(default_val);                                                                 \
  }                                                                                                                \
                                                                                                                   \
  inline ival_type Var::unwrapTo##method_variant() {                                                               \
    return this->unwrapTo<ival_type>();                                                                            \
  }

DEFINE_UNWRAP_TO(at::Tensor, Tensor)
DEFINE_UNWRAP_TO(int64_t, Int)
DEFINE_UNWRAP_TO(double, Double)
DEFINE_UNWRAP_TO(bool, Bool)
DEFINE_UNWRAP_TO(std::string, String)
DEFINE_UNWRAP_TO(c10::Scalar, Scalar)
DEFINE_UNWRAP_TO(c10::List<int64_t>, IntList)
DEFINE_UNWRAP_TO(c10::List<double>, DoubleList)
DEFINE_UNWRAP_TO(c10::List<bool>, BoolList)
DEFINE_UNWRAP_TO(c10::List<at::Tensor>, TensorList)

} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
