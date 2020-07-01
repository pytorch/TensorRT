package(default_visibility = ["//visibility:public"])

config_setting(
   name = "aarch64",
   values = {
        "cpu" : "aarch64",
   },
)

cc_library(
    name = "cublas_headers",
    hdrs = ["include/cublas.h"] + glob(["include/cublas+.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cublas_lib",
    shared_library = select({
        "aarch64" : "lib/aarch64-linux-gnu/libcublas.so.10",
        "//conditions:default" : "lib/x86_64-linux-gnu/libcublas.so.10",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cublas",
    deps = [
        "cublas_headers",
        "cublas_lib"
    ],
    visibility = ["//visibility:public"],
)
