package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {
        "cpu" : "aarch64",
    },
)

config_setting(
    name = "x86_64",
    values = {
        "cpu" : "x86_64",
    },
)

cc_library(
    name = "cudnn_headers",
    hdrs = ["include/cudnn.h"] + glob(["include/cudnn+.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "lib/aarch64-linux-gnu/libcudnn.so",
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


