load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "ci_rhel_x86_64_linux",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
        "@//toolchains/distro:ci_rhel",
    ],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "cudnn_headers",
    hdrs = ["include/cudnn.h"] + glob([
        "include/cudnn_*.h",
    ]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libcudnn.so",
        ":ci_rhel_x86_64_linux": "lib64/libcudnn.so",
        ":windows": "bin/cudnn64_7.dll",  #Need to configure specific version for windows
        "//conditions:default": "lib/x86_64-linux-gnu/libcudnn.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
    deps = [
        "cudnn_headers",
        "cudnn_lib",
    ],
)
