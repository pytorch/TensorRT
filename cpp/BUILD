load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

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
    ] + select({
        ":rtx_win": [],
        ":rtx_x86_64": [],
        "//conditions:default": [
            "src/ptq.cpp",
        ],
    }),
    hdrs = [
        "include/torch_tensorrt/logging.h",
        "include/torch_tensorrt/macros.h",
        "include/torch_tensorrt/torch_tensorrt.h",
    ] + select({
        ":rtx_win": [],
        ":rtx_x86_64": [],
        "//conditions:default": [
            "include/torch_tensorrt/ptq.h",
        ],
    }),
    linkstatic = True,
    strip_include_prefix = "include/",
    deps = [
        "//core",
        "//core/ir",
        "//core/util:prelude",
    ],
    alwayslink = True,
)

filegroup(
    name = "api_headers",
    srcs = glob(["include/**/*.h"]),
)
