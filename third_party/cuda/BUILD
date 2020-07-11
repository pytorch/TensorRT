package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    constraint_values = [
        "@platforms//cpu:aarch64",
        "@platforms//os:linux",
    ],
)

cc_library(
    name = "cudart",
    srcs = select({
        ":aarch64_linux": [
            "targets/aarch64-linux-gnu/lib/libcudart.so",
        ],
        "//conditions:default": [
            "targets/x86_64-linux/lib/libcudart.so",
        ],
    }),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
)

cc_library(
    name = "nvToolsExt",
    srcs = select({
        ":aarch64_linux": [
            "targets/aarch64-linux/lib/libnvToolsExt.so.1",
        ],
        "//conditions:default": [
            "targets/x86_64-linux/lib/libnvToolsExt.so.1",
        ],
    }),
)

cc_library(
    name = "cuda",
    srcs = select({
        ":aarch64_linux": glob([
            "targets/aarch64-linux/lib/**/lib*.so",
        ]),
        "//conditions:default": glob([
            "targets/x86_64-linux/lib/**/lib*.so",
        ]),
    }),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublas",
    srcs = glob([
        "lib/**/*libcublas.so",
    ]),
    hdrs = glob([
        "include/**/*cublas*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
