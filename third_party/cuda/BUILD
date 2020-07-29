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
    name = "cudart",
    srcs = select({
        ":aarch64_linux": [
            "targets/aarch64-linux/lib/libcudart.so",
        ],
        ":windows": [
            "lib/x64/cudart.lib",
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
        ":windows": [
            "bin/nvToolsExt64_1.dll",
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
        ":windows": [
            "bin/*.dll",
        ],
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
    srcs = select({
        ":aarch64_linux": glob([
            "lib/**/*libcublas.so",
        ]),
        ":windows": glob([
            "lib/x64/cublas.lib",
        ]),
        "//conditions:default": glob([
            "lib/**/*libcublas.so",
        ]),
    }),
    hdrs = glob([
        "include/**/*cublas*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
