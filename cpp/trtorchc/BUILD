package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {
        "cpu" : "aarch64"
	}
)

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

config_setting(
    name = "use_pre_cxx11_abi_aarch64",
    values = {
        "define": "abi=pre_cxx11_abi",
        "cpu" : "aarch64",
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
	":use_pre_cxx11_abi" : ["@libtorch_pre_cxx11_abi//:libtorch",
         			"@libtorch_pre_cxx11_abi//:caffe2"],
	":use_pre_cxx11_abi_aarch64" : ["@libtorch_pre_cxx11_abi_aarch64//:libtorch",
         			"@libtorch_pre_cxx11_abi_aarch64//:caffe2"],
	"//conditions:default": ["@libtorch//:libtorch",
			         "@libtorch//:caffe2"],
	}),
)
