load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "cudart",
    srcs = select({
        ":windows": [
            "lib/x64/cudart.lib",
        ],
        "//conditions:default": [
            "lib64/libcudart.so",
        ],
    }),
    hdrs = glob([
        "include/**/*",
    ]),
    includes = ["include/"],
)

cc_library(
    name = "nvToolsExt",
    srcs = select({
        ":windows": [
            "bin/nvToolsExt64_1.dll",
        ],
        "//conditions:default": [
            "lib64/libnvToolsExt.so.1",
        ],
    }),
)

cc_library(
    name = "cuda",
    srcs = select({
        ":windows": [
            "bin/*.dll",
        ],
        "//conditions:default": glob([
            "lib64/**/lib*.so",
        ]),
    }),
    hdrs = glob([
        "include/**/*",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublas",
    srcs = select({
        ":windows": [
            "lib/x64/cublas.lib",
        ],
        "//conditions:default": glob([
            "lib64/**/*libcublas.so",
        ]),
    }),
    hdrs = glob([
        "include/**/*",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
