package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "cifar10",
    srcs = [
        "cifar10.cpp",
    ],
    hdrs = [
        "cifar10.h",
    ],
    deps = select({
        ":use_pre_cxx11_abi": [
            "@libtorch_pre_cxx11_abi//:libtorch",
        ],
        "//conditions:default": [
            "@libtorch//:libtorch",
        ],
    }),
)
