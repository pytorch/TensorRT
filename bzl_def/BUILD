load("@rules_cc//cc:defs.bzl", "cc_library")

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
    name = "libtorchtrt",
    srcs = select({
        ":windows": [
            "lib/x64/torchtrt.dll",
        ],
        "//conditions:default": [
            "lib/libtorchtrt.so",
        ],
    }),
    hdrs = glob([
        "include/**/*.h",
    ]),
    includes = ["include/"],
    strip_include_prefix = "include",
)

cc_library(
    name = "libtorchtrt_runtime",
    srcs = select({
        ":windows": [
            "lib/x64/torchtrt_runtime.dll",
        ],
        "//conditions:default": [
            "lib/libtorchtrt_runtime.so",
        ],
    }),
)

cc_library(
    name = "libtorchtrt_plugins",
    srcs = select({
        ":windows": [
            "lib/x64/torchtrt_plugins.dll",
        ],
        "//conditions:default": [
            "lib/libtorchtrt_plugins.so",
        ],
    }),
    hdrs = glob([
        "include/torch_tensorrt/core/plugins/**/*.h",
    ]),
    includes = ["include/"],
    strip_include_prefix = "include",
)

cc_library(
    name = "torch_tensorrt_core_hdrs",
    hdrs = glob([
        "include/torch_tensorrt/core/**/*.h",
    ]),
    includes = ["include/torch_tensorrt/"],
    strip_include_prefix = "include/torch_tensorrt",
)

# Alias for ease of use
cc_library(
    name = "torch_tensorrt",
    deps = [
        ":libtorchtrt",
    ],
)
