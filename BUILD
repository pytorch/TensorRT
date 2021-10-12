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
        "//core/ir:include",
        "//core/lowering:include",
        "//core/lowering/passes:include",
        "//core/partitioning:include",
        "//core/plugins:impl_include",
        "//core/plugins:include",
        "//core/runtime:include",
        "//core/util:include",
        "//core/util/logging:include",
    ],
)

pkg_tar(
    name = "include",
    srcs = [
        "//cpp:api_headers",
    ],
    package_dir = "include/trtorch/",
)

pkg_tar(
    name = "lib",
    srcs = select({
        ":windows": ["//cpp/lib:trtorch.dll"],
        "//conditions:default": [
            "//cpp/lib:libtrtorch.so",
            "//cpp/lib:libtrtorchrt.so",
            "//cpp/lib:libtrtorch_plugins.so",
        ],
    }),
    mode = "0755",
    package_dir = "lib/",
)

pkg_tar(
    name = "bin",
    srcs = [
        "//cpp/bin/trtorchc",
    ],
    mode = "0755",
    package_dir = "bin/",
)

pkg_tar(
    name = "libtrtorch",
    srcs = [
        "//:LICENSE",
        "//bzl_def:BUILD.bzl",
        "//bzl_def:WORKSPACE"
    ],
    extension = "tar.gz",
    package_dir = "trtorch",
    deps = [
        ":lib",
        ":include",
        ":include_core",
    ] + select({
        ":windows": [],
        "//conditions:default": [":bin"],
    }),
)
