/*
 * Copyright (c) NVIDIA Corporation.
 * All rights reserved.
 *
 * This library is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#if defined(USE_CMAKE_GENERATED_EXPORT_HEADER)
#include <torch_tensorrt_export.h>
#else
#if defined(__GNUC__)
#define TORCHTRT_API __attribute__((__visibility__("default")))
#define TORCHTRT_HIDDEN __attribute__((__visibility__("hidden")))
#else
#define TORCHTRT_API
#define TORCHTRT_HIDDEN
#endif // defined(__GNUC__)
#endif // defined(USE_CMAKE_GENERATED_EXPORT_HEADER)

// Does this need to be gaurded or something?
#define XSTR(x) #x
#define STR(x) XSTR(x)

#define TORCH_TENSORRT_MAJOR_VERSION 2
#define TORCH_TENSORRT_MINOR_VERSION 6
#define TORCH_TENSORRT_PATCH_VERSION 0
#define TORCH_TENSORRT_VERSION      \
  STR(TORCH_TENSORRT_MAJOR_VERSION) \
  "." STR(TORCH_TENSORRT_MINOR_VERSION) "." STR(TORCH_TENSORRT_PATCH_VERSION)

// Setup namespace aliases for ease of use
namespace torch_tensorrt {
namespace torchscript {}
namespace ts = torchscript;
} // namespace torch_tensorrt
namespace torchtrt = torch_tensorrt;

#define TORCHTRT_DEPRICATION_WARNING_STR \
  "The torchscript frontend is now deprecated, it will be disabled in H1Y25. The Torch-TensorRT runtime will continue to support running torchscript programs, however torchscript programs should be created by using torch.jit.trace on ExportedPrograms created by the dynamo frontend"
