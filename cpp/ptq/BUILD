package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "ptq",
    srcs = [
        "main.cpp"
    ],
    deps = [
        "@libtorch//:libtorch",
        "//cpp/api:trtorch"
    ],
)
