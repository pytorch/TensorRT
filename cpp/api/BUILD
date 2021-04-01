package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trtorch",
    hdrs = [
        "include/trtorch/trtorch.h",
        "include/trtorch/logging.h",
        "include/trtorch/macros.h",
        "include/trtorch/ptq.h"
    ],
    srcs = [
        "src/compile_spec.cpp",
        "src/logging.cpp",
        "src/trtorch.cpp",
        "src/ptq.cpp"
    ],
    deps = [
        "//core",
        "//core/util:prelude",
        "//core/plugins",
    ],
    strip_include_prefix = "include/",
    linkstatic = True,
    alwayslink = True
)


filegroup(
    name = "api_headers",
    srcs = glob(["include/**/*.h"]),
)
