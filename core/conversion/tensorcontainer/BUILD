package(default_visibility = ["//visibility:public"])

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
	    "cpu": "aarch64",
    },
)
config_setting(
    name="aarch64",
    values={
        "cpu":"aarch64"
    },
)

cc_library(
    name = "tensorcontainer",
    hdrs = [
        "TensorContainer.h",
    ],
    srcs = [
        "TensorContainer.cpp",
    ],
    deps = [
        "//core/util:prelude",
    ] + select({
        ":aarch64": ["@libtorch_aarch64//:libtorch",
		     "@tensorrt_aarch64//:nvinfer"],
        ":use_pre_cxx11_abi_aarch64": ["@libtorch_pre_cxx11_abi_aarch64//:libtorch",
				       "@tensorrt_aarch64//:nvinfer"],
        ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch",
				"@tensorrt//:nvinfer"],
        "//conditions:default":  ["@libtorch//:libtorch",
				  "@tensorrt//:nvinfer"],
    }),
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/tensorcontainer/",
    srcs = [
        "TensorContainer.h",
    ],
)
