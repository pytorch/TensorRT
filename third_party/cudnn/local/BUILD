package(default_visibility = ["//visibility:public"])

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
    name = "cudnn_headers",
    hdrs = ["include/cudnn.h"] + glob(["include/cudnn+.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = select({
        ":aarch64_linux": "lib/aarch64-linux-gnu/libcudnn.so",
        ":windows": glob(["bin/cudnn64_*.dll"])[0],
        "//conditions:default": "lib/x86_64-linux-gnu/libcudnn.so",
    }),
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


