package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "trtorchexec",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "@libtorch//:libtorch",
        "//cpp/api:trtorch"
    ],
)
