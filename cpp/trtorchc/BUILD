package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {
        "cpu" : "aarch64"
	}
)

config_setting(
    name = "x86_64",
    values = {
        "cpu" : "x86_64"
	}
)

cc_binary(
    name = "trtorchc",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "//third_party/args",
        "//cpp/api:trtorch"
    ] + select({
        ":aarch64" : ["@libtorch_aarch64//:libtorch",
		      "@libtorch_aarch64//:caffe2"],
	"//conditions:default": ["@libtorch//:libtorch",
			         "@libtorch//:caffe2"],
	}),
)
