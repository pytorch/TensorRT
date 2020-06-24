package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "benchmark",
    srcs = [
        "main.cpp",
        "timer.h"
    ],
    deps = [
    ] = select([
        ":aarch64": ["@libtorch_aarch64//:libtorch"],
	     "//conditions:default": ["@libtorch//:libtorch",
			                      "@libtorch//:caffe2",
			                      "//cpp/api:trtorch"
			                     ],
	]),
)
