load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl"
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack"
    },
)
config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_binary(
    name = "torchtrt_aoti_example",
    srcs = [
        "inference.cpp"
    ],
    linkopts = [
        "-ldl",
    ],
    deps = [
        "//cpp:torch_tensorrt",
    ] + select({
        ":windows": [
            "@libtorch_win//:caffe2",
            "@libtorch_win//:libtorch"
        ],
        ":use_torch_whl": [
            "@torch_whl//:caffe2",
            "@torch_whl//:libtorch"
        ],
        ":jetpack": [
            "@torch_l4t//:caffe2",
            "@torch_l4t//:libtorch"
        ],
        "//conditions:default": [
            "@libtorch",
            "@libtorch//:caffe2",
        ],
    }),
)
