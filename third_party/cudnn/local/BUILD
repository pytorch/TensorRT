package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudnn_headers",
    hdrs = ["include/cudnn.h"] + glob(["include/cudnn+.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "lib/x86_64-linux-gnu/libcudnn.so.7.6.5",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    deps = [
        "cudnn_headers",
        "cudnn_lib"
    ],
    visibility = ["//visibility:public"],
)


