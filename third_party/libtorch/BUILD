package(default_visibility = ["//visibility:public"])

cc_library(
    name = "libtorch",
    deps = [
        ":torch",
    ],
)

cc_library(
    name = "torch",
    srcs = [
        "lib/libtorch.so",
        "lib/libtorch_cpu.so",
        "lib/libtorch_cuda.so",
        "lib/libtorch_global_deps.so",
    ],
    hdrs = glob(
        [
            "include/torch/**/*.h",
        ],
        exclude = [
            "include/torch/csrc/api/include/**/*.h",
        ],
    ) + glob([
        "include/torch/csrc/api/include/**/*.h",
    ]),
    includes = [
        "include",
        "include/torch/csrc/api/include/",
    ],
    deps = [
        ":ATen",
        ":c10_cuda",
    ],
)

cc_library(
    name = "c10_cuda",
    srcs = ["lib/libc10_cuda.so"],
    hdrs = glob([
        "include/c10/**/*.h",
    ]),
    strip_include_prefix = "include",
    deps = [
        ":c10",
    ],
)

cc_library(
    name = "c10",
    srcs = ["lib/libc10.so"],
    hdrs = glob([
        "include/c10/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "ATen",
    hdrs = glob([
        "include/ATen/**/*.h",
    ]),
    strip_include_prefix = "include",
)

cc_library(
    name = "caffe2",
    srcs = [
        "lib/libcaffe2_detectron_ops_gpu.so",
        "lib/libcaffe2_module_test_dynamic.so",
        "lib/libcaffe2_nvrtc.so",
        "lib/libcaffe2_observers.so",
    ],
    hdrs = glob([
        "include/caffe2/**/*.h",
    ]),
    strip_include_prefix = "include",
)
