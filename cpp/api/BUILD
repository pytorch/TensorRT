package(default_visibility = ["//visibility:public"])

cc_library(
    name = "trtorch",
    hdrs = [
        "include/trtorch/trtorch.h",
        "include/trtorch/logging.h",
        "include/trtorch/macros.h"
    ],
    srcs = [
        "src/trtorch.cpp",
        "src/extra_info.cpp",
        "src/logging.cpp"
    ],
    deps = [
        "//core",
        "//core/util:prelude"
    ],
    strip_include_prefix = "include/",
    linkstatic = True 
)
    
    
filegroup(
    name = "api_headers",
    srcs = glob(["include/**/*.h"]),
)
