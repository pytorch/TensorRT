package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libtorchtrt_plugins.so",
    srcs = [],
    deps = [
        "//core/plugins:torch_tensorrt_plugins"
    ],
    linkstatic = True,
    linkshared = True
)

cc_binary(
    name = "libtorchtrt_runtime.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/runtime:runtime",
        "//core/plugins:torch_tensorrt_plugins"
    ],
)

cc_binary(
    name = "libtorchtrt.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp:torch_tensorrt",
    ],
)

cc_binary(
    name = "torch_tensorrt.dll",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp:torch_tensorrt",
    ],
)
