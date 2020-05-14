package(default_visibility = ["//visibility:public"])

config_setting(
    name = "dont_use_cxx11_abi",
    values = {
        "define": "api=python",
    }
)

cc_library(
    name = "execution",
    hdrs = [
        "execution.h",
    ],
    srcs = [
        "TRTEngine.cpp",
        "TRTEngineManager.cpp",
        "register_trt_op.cpp",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude"
    ] + select({
        ":dont_use_cxx11_abi":  ["@libtorch_non_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/execution/",
    srcs = ["execution.h"],
)
