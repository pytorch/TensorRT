package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
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
        ":aarch64_linux": "lib/aarch64-linux-gnu/libcublas.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libcublas.so",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cublas_lt_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libcublasLt.so",
        "//conditions:default": "lib/x86_64-linux-gnu/libcublasLt.so",
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cublas",
    visibility = ["//visibility:public"],
    deps = [
        "cublas_headers",
        "cublas_lib",
        "cublas_lt_lib",
    ],
)