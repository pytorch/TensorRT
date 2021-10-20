package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "libtrtorch",
    srcs = select({
    ":windows": [
        "lib/x64/trtorch.dll",
    ],
    "//conditions:default": [
        "lib/libtrtorch.so",
    ],
    }),
    hdrs = glob([
        "include/**/*.h",
    ]),
    strip_include_prefix = "include",
    includes = ["include/"]
)

cc_library(
    name = "libtrtorchrt",
    srcs = select({
        ":windows": [
            "lib/x64/trtorchrt.dll"
        ],
        "//conditions:default": [
            "lib/libtrtorchrt.so"
        ]
    })
)

cc_library(
    name = "libtrtorch_plugins",
    srcs = select({
        ":windows": [
            "lib/x64/trtorch_plugins.dll"
        ],
        "//conditions:default": [
            "lib/libtrtorch_plugins.so"
        ]
    }),
    hdrs = glob([
        "include/trtorch/core/plugins/**/*.h",
    ]),
    strip_include_prefix = "include",
    includes = ["include/"]
)

cc_library(
    name = "trtorch_core_hdrs",
    hdrs = glob([
        "include/trtorch/core/**/*.h"
    ]),
    strip_include_prefix = "include/trtorch",
    includes = ["include/trtorch/"]
)

# Alias for ease of use
cc_library(
    name = "trtorch",
    deps = [
        ":libtrtorch",
    ]
)