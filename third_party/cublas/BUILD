package(default_visibility = ["//visibility:public"])

# NOTE: This BUILD file is only really targeted at aarch64, the rest of the configuration is just to satisfy bazel, x86 uses the cublas source from the CUDA build file since it will be versioned with CUDA.

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "cublas_headers",
    hdrs = select({
        ":aarch64_linux": ["include/cublas.h"] + glob(["usr/include/cublas+.h"]),
        "//conditions:default": ["local/cuda/include/cublas.h"] + glob(["usr/cuda/include/cublas+.h"]),
    }),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cublas_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libcublas.so",
        ":windows": "lib/x64/cublas.lib",
        "//conditions:default": "local/cuda/targets/x86_64-linux/lib/libcublas.so",
    }),
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cublas_lt_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libcublasLt.so",
        "//conditions:default": "local/cuda/targets/x86_64-linux/lib/libcublasLt.so",
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