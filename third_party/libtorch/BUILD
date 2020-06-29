package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64",
    values = {
	"cpu" : "aarch64"
    },
)

cc_library(
    name = "libtorch",
    deps = [
        ":torch"
    ]
)

cc_library(
    name = 'torch',
    hdrs = ['include'] + glob(
        [
            'include/torch/**/*.h',
        ], exclude = [
            'include/torch/csrc/api/include/**/*.h'
        ]
    ) + glob([
        'include/torch/csrc/api/include/**/*.h'
    ]),
    srcs = [
        'lib/libtorch.so',
        'lib/libtorch_cuda.so',
        'lib/libtorch_cpu.so',
        'lib/libtorch_global_deps.so',
    ],
    deps = [
        ":ATen",
        ":c10_cuda",
    ],
    includes = [
        "include",
        "include/torch/csrc/api/include/"
    ]
)

cc_library(
    name = 'c10_cuda',
    hdrs = glob([
        'include/c10/**/*.h'
    ]),
    srcs = ["lib/libc10_cuda.so"],
    strip_include_prefix = "include",
    deps = [
        ":c10"
    ]
)

cc_library(
    name = 'c10',
    hdrs = glob([
        'include/c10/**/*.h'
    ]),
    srcs = ["lib/libc10.so"],
    strip_include_prefix = "include",
)

cc_library(
    name = "ATen",
    hdrs = glob([
        "include/ATen/**/*.h"
    ]),
    strip_include_prefix = "include"
)

cc_library(
    name = 'caffe2',
    hdrs = glob([
        'include/caffe2/**/*.h'
    ]),
    srcs = [
        'lib/libcaffe2_nvrtc.so',
        'lib/libcaffe2_detectron_ops_gpu.so',
        'lib/libcaffe2_observers.so',
        'lib/libcaffe2_module_test_dynamic.so'
    ],
    strip_include_prefix = "include",
)
