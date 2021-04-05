package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libtrtorch_plugins.so",
    srcs = [],
    deps = [
        "//core/plugins:trtorch_plugins"
    ],
    linkstatic = True,
    linkshared = True
)

cc_binary(
    name = "libtrtorchrt.so",
    srcs = [],
    deps = [
        "//core/runtime:runtime",
        "//core/plugins:trtorch_plugins"
    ],
    linkstatic = True,
    linkshared = True
)

cc_binary(
    name = "libtrtorch.so",
    srcs = [],
    deps = [
        "//cpp/api:trtorch"
    ],
    linkstatic = True,
    linkshared = True
)

cc_binary(
    name = "trtorch.dll",
    srcs = [],
    deps = [
        "//cpp/api:trtorch"
    ],
    linkstatic = True,
    linkshared = True
)
