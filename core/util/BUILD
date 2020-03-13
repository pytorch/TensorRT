package(default_visibility = ["//visibility:public"])

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
    deps = [
        "@libtorch//:libtorch",
    ]
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
        "@libtorch//:libtorch",
        "@tensorrt//:nvinfer"
    ]
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
        "@libtorch//:libtorch",
        "@tensorrt//:nvinfer",
        "//core/util/logging",
        ":macros"
    ]
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
