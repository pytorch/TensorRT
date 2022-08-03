load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "libtorch",
    deps = [
        ":torch",
    ],
)

cc_library(
    name = "torch",
    srcs = select({
        ":windows": [
            "lib/torch.lib",
            "lib/torch_cuda.lib",
            "lib/torch_cpu.lib",
            "lib/torch_global_deps.dll",
        ],
        "//conditions:default": [
            "lib/libtorch.so",
            "lib/libtorch_cuda.so",
            "lib/libtorch_cpu.so",
            "lib/libtorch_global_deps.so",
        ],
    }),
    hdrs = glob(
        [
            "include/torch/**/*.h",
        ],
        exclude = [
            "include/torch/csrc/api/include/**/*.h",
        ],
    ) + glob([
        "include/torch/csrc/api/include/**/*.h",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include/",
    ],
    deps = [
        ":ATen",
        ":c10_cuda",
    ],
)

cc_library(
    name = "c10_cuda",
    srcs = select({
        ":windows": ["lib/c10_cuda.lib"],
        "//conditions:default": ["lib/libc10_cuda.so"],
    }),
    hdrs = glob([
        "include/c10/**/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":c10",
    ],
)

cc_library(
    name = "c10",
    srcs = select({
        ":windows": ["lib/c10.lib"],
        "//conditions:default": ["lib/libc10.so"],
    }),
    hdrs = glob([
        "include/c10/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "ATen",
    hdrs = glob([
        "include/ATen/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "caffe2",
    srcs = select({
        ":windows": [
            "lib/caffe2_nvrtc.lib",
        ],
        "//conditions:default": [
            "lib/libcaffe2_nvrtc.so",
        ],
    }),
    hdrs = glob([
        "include/caffe2/**/*.h",
    ]),
    strip_include_prefix = "include",
)
