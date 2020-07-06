package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "tensorcontainer",
    srcs = [
        "TensorContainer.cpp",
    ],
    hdrs = [
        "TensorContainer.h",
    ],
    deps = [
        "//core/util:prelude",
        "@tensorrt//:nvinfer",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    srcs = [
        "TensorContainer.h",
    ],
    package_dir = "core/conversion/tensorcontainer/",
)
