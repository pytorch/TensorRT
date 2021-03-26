load("@rules_pkg//:pkg.bzl", "pkg_tar")

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

pkg_tar(
    name = "include_core",
    package_dir = "include/trtorch",
    deps = [
        "//core:include",
        "//core/conversion:include",
        "//core/conversion/conversionctx:include",
        "//core/conversion/converters:include",
        "//core/conversion/var:include",
        "//core/conversion/tensorcontainer:include",
        "//core/conversion/evaluators:include",
        "//core/plugins:include",
        "//core/runtime:include",
        "//core/lowering:include",
        "//core/lowering/passes:include",
        "//core/util:include",
        "//core/util/logging:include"
    ],
)

pkg_tar(
    name = "include",
    package_dir = "include/trtorch/",
    srcs = [
        "//cpp/api:api_headers",
    ],
)

pkg_tar(
    name = "lib",
    package_dir = "lib/",
    srcs = select({
        ":windows": ["//cpp/api/lib:trtorch.dll"],
        "//conditions:default": [
            "//cpp/api/lib:libtrtorch.so",
            "//cpp/api/lib:libtrtorchrt.so",
            "//cpp/api/lib:libtrtorch_plugins.so",
        ],
    }),
    mode = "0755",
)


pkg_tar(
    name = "bin",
    package_dir = "bin/",
    srcs = [
        "//cpp/trtorchc:trtorchc",
    ],
    mode = "0755",
)


pkg_tar(
    name = "libtrtorch",
    extension = "tar.gz",
    package_dir = "trtorch",
    srcs = [
        "//:LICENSE"
    ],
    deps = [
        ":lib",
        ":include",
        ":include_core",
    ] + select({
        ":windows": [],
        "//conditions:default": [":bin"],
    }),
)
