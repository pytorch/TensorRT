load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "windows_arm64",
    constraint_values = [
        "@platforms//cpu:arm64",
        "@platforms//os:windows",
    ],
)

# Every header in the toolkit, and the only place the glob is written: the library
# targets below take their headers from here. One catch-all pattern rather than a
# list of extensions, because `include/**/*` already matches every .h, .hpp and
# .inl under it. Keeping the headers in a target of their own is the shape
# third_party/cublas and third_party/cudnn/archive already use.
cc_library(
    name = "cuda_headers",
    hdrs = glob(["include/**/*"]),
    includes = ["include/"],
)

cc_library(
    name = "cudart",
    srcs = select({
        ":windows": [
            "lib/x64/cudart.lib",
        ],
        ":windows_arm64": [
            "lib/arm64/cudart.lib",
        ],
        "//conditions:default": [
            "lib64/libcudart.so",
        ],
    }),
    deps = [":cuda_headers"],
)

cc_library(
    name = "nvToolsExt",
    srcs = select({
        ":windows": [
            "bin/nvToolsExt64_1.dll",
        ],
        ":windows_arm64": [
            "bin/arm64/nvToolsExt64_1.dll",
        ],
        "//conditions:default": [
            "lib64/libnvToolsExt.so.1",
        ],
    }),
)

cc_library(
    name = "cuda",
    srcs = select({
        ":windows": [
            "bin/*.dll",
        ],
        ":windows_arm64": glob(
            ["bin/arm64/*.dll"],
            allow_empty = True,
        ),
        "//conditions:default": glob(
            [
                "lib64/**/lib*.so",
            ],
            allow_empty = True,
        ),
    }),
    linkopts = ["-Wl,-rpath,lib/"],
    deps = [":cuda_headers"],
)

cc_library(
    name = "cublas",
    srcs = select({
        ":windows": [
            "lib/x64/cublas.lib",
        ],
        ":windows_arm64": [
            "lib/arm64/cublas.lib",
        ],
        "//conditions:default": glob(
            [
                "lib64/**/*libcublas.so",
            ],
            allow_empty = True,
        ),
    }),
    linkopts = ["-Wl,-rpath,lib/"],
    deps = [":cuda_headers"],
)
