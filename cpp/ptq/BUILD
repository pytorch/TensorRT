package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ptq",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//cpp/ptq/benchmark",
        "//cpp/ptq/datasets:cifar10",
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "//cpp/api:trtorch",
    ] + select([
	":aarch64" : ["@tensorrt_aarch64//:nvinfer"],
	"//conditions:default" : ["@tensorrt//:nvinfer"],
		    ]),
    copts = [
        "-pthread"
    ],
    linkopts = [
        "-lpthread",
    ]
)
