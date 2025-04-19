load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_pkg//:pkg.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

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
        "Platform.cpp",
        "RTDevice.cpp",
        "TRTEngine.cpp",
        "TRTEngineProfiler.cpp",
        "execute_engine.cpp",
        "register_jit_hooks.cpp",
        "runtime.cpp",
    ],
    hdrs = [
        "Platform.h",
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
        ":windows": [
            "@libtorch_win//:libtorch",
            "@tensorrt_win//:nvinfer",
        ],
        "//conditions:default": [
            "@libtorch",
            "@tensorrt//:nvinfer",
        ],
    }),
    alwayslink = True,
)

pkg_tar(
    name = "include",
    srcs = [
        "Platform.h",
        "RTDevice.h",
        "TRTEngine.h",
        "TRTEngineProfiler.h",
        "runtime.h",
    ],
    package_dir = "core/runtime/",
)
