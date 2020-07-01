package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

cc_binary(
    name = "ptq",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//cpp/ptq/benchmark",
        "//cpp/ptq/datasets:cifar10",
        "//cpp/api:trtorch",
	"@tensorrt//:nvinfer"
    ] + select({
        ":use_pre_cxx11_abi": [
            "@libtorch_pre_cxx11_abi//:libtorch",
            "@libtorch_pre_cxx11_abi//:caffe2",
        ],
        "//conditions:default": [
            "@libtorch//:libtorch",
            "@libtorch//:caffe2",
        ]
    }), 
    copts = [
        "-pthread"
    ],
    linkopts = [
        "-lpthread",
    ]
)
