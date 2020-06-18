package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "trtorchc",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "@libtorch//:libtorch",
        "@libtorch//:caffe2",
        "//third_party/args",
        "//cpp/api:trtorch"
    ],
)
