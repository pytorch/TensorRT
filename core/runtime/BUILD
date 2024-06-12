load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "runtime",
    srcs = [
        "DeviceList.cpp",
        "RTDevice.cpp",
        "TRTEngine.cpp",
        "TRTEngineProfiler.cpp",
        "execute_engine.cpp",
        "register_jit_hooks.cpp",
        "runtime.cpp",
    ],
    hdrs = [
        "RTDevice.h",
        "TRTEngine.h",
        "TRTEngineProfiler.h",
        "runtime.h",
    ],
    linkopts = [
        "-lstdc++fs",
    ],
    deps = [
        "//core/plugins:torch_tensorrt_plugins",
        "//core/util:prelude",
    ] + select({
        ":windows": ["@tensorrt_win//:nvinfer"],
        "//conditions:default": ["@tensorrt//:nvinfer"],
    }) + select({
        ":windows": ["@libtorch_win//:libtorch"],
        ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default": ["@libtorch"],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "RTDevice.h",
        "TRTEngine.h",
        "TRTEngineProfiler.h",
        "runtime.h",
    ],
    package_dir = "core/runtime/",
)
