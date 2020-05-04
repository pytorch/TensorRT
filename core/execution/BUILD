package(default_visibility = ["//visibility:public"])

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
        "@libtorch//:libtorch",
        "//core/util:prelude"
    ],
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/execution/",
    srcs = ["execution.h"],
)
