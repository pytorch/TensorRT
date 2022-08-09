load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "runtime",
    srcs = [
        "CudaDevice.cpp",
        "DeviceList.cpp",
        "TRTEngine.cpp",
        "execute_engine.cpp",
        "register_jit_hooks.cpp",
        "runtime.cpp",
    ],
    hdrs = [
        "runtime.h",
    ],
    deps = [
        "@tensorrt//:nvinfer",
        "//core/util:prelude",
        "//core/plugins:torch_tensorrt_plugins",
    ] + select({
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch//:libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = ["runtime.h"],
    package_dir = "core/runtime/",
)
