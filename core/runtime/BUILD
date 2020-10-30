package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_library(
    name = "runtime",
    hdrs = [
        "runtime.h",
    ],
    srcs = [
        "TRTEngine.cpp",
        "register_trt_op.cpp",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude"
    ] + select({
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/runtime/",
    srcs = ["runtime.h"],
)
