package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libtrtorchrt.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/runtime",
    ],
)

cc_binary(
    name = "libtrtorch.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp/api:trtorch",
    ],
)

cc_binary(
    name = "trtorch.dll",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp/api:trtorch",
    ],
)
