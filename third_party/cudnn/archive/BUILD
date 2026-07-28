load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudnn_headers",
    hdrs = glob(["include/cudnn*.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "lib/libcudnn.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
    deps = [
        "cudnn_headers",
        "cudnn_lib",
    ],
)
