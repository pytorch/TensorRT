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
    name = "rtx_win",
    constraint_values = [
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
    name = "tensorrt_executorch_binding_names",
    srcs = [
        "src/torch_tensorrt/executorch/TensorRTBindingNames.cpp",
    ],
    hdrs = [
        "src/torch_tensorrt/executorch/TensorRTBindingNames.h",
    ],
    strip_include_prefix = "src/torch_tensorrt/executorch",
)

cc_library(
    name = "tensorrt_executorch_backend",
    srcs = [
        "src/torch_tensorrt/executorch/TensorRTBackend.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/executorch/TensorRTBackend.h",
    ],
    # Build the TensorRT backend as a static library. The final application
    # links this target together with the ExecuTorch runtime it was compiled
    # against, avoiding any runtime plugin/dlopen dependency.
    target_compatible_with = select({
        ":linux_x86_64": [],
        ":sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    strip_include_prefix = "include",
    deps = [
        ":tensorrt_executorch_binding_names",
        ":tensorrt_executorch_blob_header",
    ] + select({
        ":linux_x86_64": [
            "@executorch//:executorch_headers",
            "@cuda//:cudart",
            "@tensorrt//:nvinfer",
        ],
        ":sbsa": [
            "@executorch//:executorch_headers",
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
        "src/torch_tensorrt/executorch/TensorRTBindingNames.cpp",
        "src/torch_tensorrt/executorch/TensorRTBindingNames.h",
        "src/torch_tensorrt/executorch/TensorRTBackend.cpp",
        "src/torch_tensorrt/executorch/TensorRTBlobHeader.cpp",
    ],
)

filegroup(
    name = "api_headers",
    srcs = glob(
        ["include/**/*.h"],
        exclude = ["include/torch_tensorrt/executorch/**/*.h"],
    ),
)

filegroup(
    name = "executorch_api_headers",
    srcs = [
        "include/torch_tensorrt/executorch/TensorRTBackend.h",
        "include/torch_tensorrt/executorch/TensorRTBlobHeader.h",
    ],
)
