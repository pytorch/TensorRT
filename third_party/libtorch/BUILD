package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libtorch",
    deps = [
        ":torch"
    ]
)

cc_library(
    name = 'torch',
    hdrs = glob([
        'include/torch/**/*.h',
    ],
        exclude = ['include/torch/csrc/api/include/**/*.h']
    ) + glob([
        'include/torch/csrc/api/include/**/*.h'
    ]),
    srcs = ['lib/libtorch.so'],
    strip_include_prefix = "include",
    deps = [
        ":ATen",
        ":torch_deps",
        ":c10_cuda",
        #"@cuda//:cudart",
        #"@cuda//:nvToolsExt"
    ]
)

cc_library(
    name = 'torch_deps',
    srcs = [
        'lib/libnvToolsExt-3965bdd0.so.1',
        "lib/libcudart-1b201d85.so.10.1",
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
    deps = [
        ":c10_deps",
        "@cuda//:cudart",
    ]
)

cc_library(
    name = "c10_deps",
    srcs = [
        "lib/libgomp-753e6e92.so.1",
        "lib/libcudart-1b201d85.so.10.1",
    ]
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
    srcs = ['lib/libcaffe2_nvrtc.so'],
    strip_include_prefix = "include",
    deps = [
        ":caffe2_deps"
    ]
)

cc_library(
    name = 'caffe2_deps',
    srcs = [
        'lib/libnvToolsExt-3965bdd0.so.1',
        'lib/libgomp-753e6e92.so.1'
    ]
)
