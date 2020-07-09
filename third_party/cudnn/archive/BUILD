package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudnn_headers",
    hdrs = glob(["include/cudnn*.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "lib64/libcudnn.so",
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


