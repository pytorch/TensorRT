load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "rtx_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "rtx_sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "rtx_win_x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

config_setting(
    name = "rtx_win_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:windows",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "rtx",
    },
)

cc_library(
    name = "torch_tensorrt",
    srcs = [
        "src/compile_spec.cpp",
        "src/logging.cpp",
        "src/torch_tensorrt.cpp",
        "src/types.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/logging.h",
        "include/torch_tensorrt/macros.h",
        "include/torch_tensorrt/torch_tensorrt.h",
    ],
    linkstatic = True,
    strip_include_prefix = "include/",
    deps = [
        "//core",
        "//core/ir",
        "//core/util:prelude",
    ],
    alwayslink = True,
)

cc_library(
    name = "tensorrt_executorch_blob_header",
    srcs = [
        "src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBlobHeader.h",
    ],
    strip_include_prefix = "include",
)

cc_library(
    name = "tensorrt_executorch_weight_streaming_budget",
    srcs = [
        "src/torch_tensorrt/executorch/WeightStreamingBudget.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/WeightStreamingBudget.h",
    ],
    strip_include_prefix = "include",
)

cc_library(
    name = "tensorrt_executorch_binding_names",
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBindingNames.h",
    ],
    strip_include_prefix = "include",
    deps = [
        "//core/runtime:tensorrt_binding_names",
    ],
)

# Registration only, kept out of :tensorrt_executorch_backend so that linking
# the backend into both a shared library and an executable cannot register the
# allocator twice. Executables that run a TensorRT delegated program depend on
# this directly.
#
# Not part of :executorch_backend_source_files. The CMake build of the reference
# runner enables the CUDA/AOTI delegate, which already registers this allocator.
cc_library(
    name = "tensorrt_executorch_cuda_device_allocator",
    srcs = [
        "src/torch_tensorrt/executorch/RegisterCudaDeviceAllocator.cpp",
    ],
    alwayslink = True,
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": [
            "@executorch//:executorch_cuda_allocator",
            "@executorch//:executorch_headers",
        ],
        ":sbsa": [
            "@executorch//:executorch_cuda_allocator",
            "@executorch//:executorch_headers",
        ],
        "//conditions:default": [],
    }),
)

# Registration only, for the same reason as the allocator above: the kernel
# registry aborts when the same kernel is registered twice, so this cannot ride
# along with the backend. Executables that run a TensorRT delegated program
# depend on this directly.
#
# Not part of :executorch_backend_source_files. The CMake build of the reference
# runner links executorch::kernels, which already registers these.
cc_library(
    name = "tensorrt_executorch_device_copy_kernels",
    srcs = [
        "src/torch_tensorrt/executorch/RegisterDeviceCopyKernels.cpp",
    ],
    alwayslink = True,
    deps = [
        "@executorch//:executorch_device_copy_kernels",
        "@executorch//:executorch_headers",
    ],
)

# Implementation detail of :tensorrt_executorch_backend, so the header sits under
# src/ and ships with the sources rather than in the installed API set. A caller
# turning the pool on needs only the option key, which is in TensorRTBackend.h.
cc_library(
    name = "tensorrt_executorch_shared_scratch_pool",
    hdrs = [
        "src/torch_tensorrt/executorch/SharedScratchPool.h",
    ],
    strip_include_prefix = "src",
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        ":linux_x86_64": [
            "@cuda//:cuda_headers",
        ],
        ":sbsa": [
            "@cuda//:cuda_headers",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "tensorrt_executorch_backend",
    srcs = [
        "src/torch_tensorrt/executorch/TensorRTBackend.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBackend.h",
    ],
    strip_include_prefix = "include",
    # Build the TensorRT backend as a static library. The final application
    # links this target together with the ExecuTorch runtime it was compiled
    # against, avoiding any runtime plugin/dlopen dependency.
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":tensorrt_executorch_binding_names",
        ":tensorrt_executorch_blob_header",
        ":tensorrt_executorch_shared_scratch_pool",
        ":tensorrt_executorch_weight_streaming_budget",
    ] + select({
        ":linux_x86_64": [
            "@executorch//:executorch_headers",
            "@executorch//:extension_cuda",
            "@cuda//:cudart",
            "@tensorrt//:nvinfer",
        ],
        ":sbsa": [
            "@executorch//:executorch_headers",
            "@executorch//:extension_cuda",
            "@cuda//:cudart",
            "@tensorrt_sbsa//:nvinfer",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

filegroup(
    name = "executorch_backend_source_files",
    srcs = [
        "src/torch_tensorrt/executorch/CMakeLists.txt",
        "src/torch_tensorrt/executorch/README.md",
        "src/torch_tensorrt/executorch/SharedScratchPool.h",
        "src/torch_tensorrt/executorch/TensorRTBackend.cpp",
        "src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp",
        "src/torch_tensorrt/executorch/WeightStreamingBudget.cpp",
    ],
)

filegroup(
    name = "api_headers",
    srcs = glob(
        ["include/**/*.h"],
        exclude = [
            "include/torch_tensorrt/executorch/**/*.h",
        ],
    ),
)

filegroup(
    name = "executorch_api_headers",
    srcs = [
        "include/torch_tensorrt/executorch/TensorRTBackend.h",
        "include/torch_tensorrt/executorch/TensorRTBindingNames.h",
        "include/torch_tensorrt/executorch/TensorRTBlobHeader.h",
        "include/torch_tensorrt/executorch/WeightStreamingBudget.h",
    ],
)
