#pragma once

#include "torch_tensorrt/serialization/TensorRTBindingNames.h"

namespace torch_tensorrt {
namespace executorch_backend {
namespace detail {

using ::torch_tensorrt::serialization::all_binding_names_present;
using ::torch_tensorrt::serialization::append_binding_name;
using ::torch_tensorrt::serialization::infer_engine_binding_names;
using ::torch_tensorrt::serialization::kBindingNameDelimiter;
using ::torch_tensorrt::serialization::parse_binding_index;
using ::torch_tensorrt::serialization::serialize_binding_names;
using ::torch_tensorrt::serialization::split_serialized_binding_names;
using ::torch_tensorrt::serialization::TensorRTBindingNames;

} // namespace detail
} // namespace executorch_backend
} // namespace torch_tensorrt
