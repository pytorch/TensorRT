load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "torch_tensorrt",
    srcs = [
        "src/compile_spec.cpp",
        "src/logging.cpp",
        "src/ptq.cpp",
        "src/torch_tensorrt.cpp",
        "src/types.cpp",
    ],
    hdrs = [
        "include/torch_tensorrt/logging.h",
        "include/torch_tensorrt/macros.h",
        "include/torch_tensorrt/ptq.h",
        "include/torch_tensorrt/torch_tensorrt.h",
    ],
    linkstatic = True,
    strip_include_prefix = "include/",
    deps = [
        "//core",
        "//core/ir",
        "//core/util:prelude",
    ],
    alwayslink = True,
)

filegroup(
    name = "api_headers",
    srcs = glob(["include/**/*.h"]),
)
