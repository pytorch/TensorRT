package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trtorch",
    srcs = [
        "src/compile_spec.cpp",
        "src/logging.cpp",
        "src/ptq.cpp",
        "src/trtorch.cpp",
    ],
    hdrs = [
        "include/trtorch/logging.h",
        "include/trtorch/macros.h",
        "include/trtorch/ptq.h",
        "include/trtorch/trtorch.h",
    ],
    linkstatic = True,
    strip_include_prefix = "include/",
    deps = [
        "//core",
        "//core/util:prelude",
    ],
    alwayslink = True,
)

filegroup(
    name = "api_headers",
    srcs = glob(["include/**/*.h"]),
)
