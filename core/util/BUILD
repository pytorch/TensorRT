package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

config_setting(
    name = "use_pre_cxx11_abi_aarch64",
    values = {
        "define": "abi=pre_cxx11_abi",
        "cpu" : "aarch64",
    }
)

config_setting(
    name="aarch64",
    values={
        "cpu":"aarch64"
    },
)

cc_library(
    name = "prelude",
    hdrs = [
        "prelude.h",
    ],
    deps = [
        "//core/util/logging",
        ":build_info",
        ":jit_util",
        ":trt_util",
        ":macros",
        ":exception"
    ]
)

cc_library(
    name = "jit_util",
    hdrs = [
        "jit_util.h",
    ],
    deps = select({
	":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        ":use_pre_cxx11_abi_aarch64":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }),
)

cc_library(
    name = "macros",
    hdrs = [
        "macros.h",
    ],
    deps = [
        ":exception"
    ]
)

cc_library(
    name = "exception",
    hdrs = [
        "Exception.h",
    ],
    srcs = [
        "Exception.cpp"
    ]
)

cc_library(
    name = "build_info",
    hdrs = [
        "build_info.h",
    ],
    deps = [
    ] + select({
        ":use_pre_cxx11_abi_aarch64": ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
        ":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }) + select({
        ":aarch64": ["@tensorrt_aarch64//:nvinfer"],
        "//conditions:default":  ["@tensorrt//:nvinfer"]
    })
)

cc_library(
    name = "trt_util",
    hdrs = [
        "trt_util.h",
    ],
    srcs = [
        "trt_util.cpp"
    ],
    deps = [
        "//core/util/logging",
        ":macros"
    ] + select({
        ":use_pre_cxx11_abi_aarch64": ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
        ":aarch64": ["@libtorch_aarch64//:libtorch"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
        "//conditions:default":  ["@libtorch//:libtorch"],
    }) + select({
        ":aarch64": ["@tensorrt_aarch64//:nvinfer"],
        "//conditions:default":  ["@tensorrt//:nvinfer"]
    })
)


load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/util/",
    srcs = [
        "//core/util:build_info.h",
        "//core/util:macros.h",
        "//core/util:Exception.h",
        "//core/util:prelude.h",
        "//core/util:jit_util.h",
        "//core/util:trt_util.h"
    ],
)
