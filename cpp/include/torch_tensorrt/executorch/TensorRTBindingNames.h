/**
 * @file TensorRTBindingNames.h
 * @brief Re-exports the shared binding-name helpers into the ExecuTorch backend.
 *
 * The delegate has to map ExecuTorch's positional arguments onto TensorRT's
 * named bindings, and the rules for that are shared with the standard runtime.
 * Rather than restate them, this header pulls
 * torch_tensorrt::core::runtime's helpers in, so both runtimes agree by
 * construction. The include is spelled two ways because the installed tarball
 * and the in-repo build lay the core runtime out differently.
 *
 * Everything here lives in a `detail` namespace: it is shared with the standard
 * runtime, not part of the ExecuTorch backend's public API.
 */
#pragma once

#if __has_include("torch_tensorrt/core/runtime/TensorRTBindingNames.h")
#include "torch_tensorrt/core/runtime/TensorRTBindingNames.h"
#else
#include "core/runtime/TensorRTBindingNames.h"
#endif

namespace torch_tensorrt {
namespace executorch_backend {
namespace detail {

using ::torch_tensorrt::core::runtime::all_binding_names_present;
using ::torch_tensorrt::core::runtime::append_binding_name;
using ::torch_tensorrt::core::runtime::infer_engine_binding_names;
using ::torch_tensorrt::core::runtime::kBindingNameDelimiter;
using ::torch_tensorrt::core::runtime::parse_binding_index;
using ::torch_tensorrt::core::runtime::serialize_binding_names;
using ::torch_tensorrt::core::runtime::split_serialized_binding_names;
using ::torch_tensorrt::core::runtime::TensorRTBindingNames;

} // namespace detail
} // namespace executorch_backend
} // namespace torch_tensorrt
