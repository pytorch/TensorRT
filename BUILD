load("@rules_pkg//:pkg.bzl", "pkg_tar")

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

pkg_tar(
    name = "include_core",
    package_dir = "include/torch_tensorrt",
    deps = [
        "//core:include",
        "//core/conversion:include",
        "//core/conversion/conversionctx:include",
        "//core/conversion/converters:include",
        "//core/conversion/evaluators:include",
        "//core/conversion/tensorcontainer:include",
        "//core/conversion/var:include",
        "//core/ir:include",
        "//core/lowering:include",
        "//core/lowering/passes:include",
        "//core/partitioning:include",
        "//core/partitioning/partitioningctx:include",
        "//core/partitioning/partitioninginfo:include",
        "//core/partitioning/segmentedblock:include",
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
    package_dir = "include/torch_tensorrt/",
)

pkg_tar(
    name = "lib",
    srcs = select({
        ":windows": ["//cpp/lib:torch_tensorrt.dll"],
        "//conditions:default": [
            "//cpp/lib:libtorchtrt.so",
            "//cpp/lib:libtorchtrt_plugins.so",
            "//cpp/lib:libtorchtrt_runtime.so",
        ],
    }),
    mode = "0755",
    package_dir = "lib/",
)

pkg_tar(
    name = "bin",
    srcs = [
        "//cpp/bin/torchtrtc",
    ],
    mode = "0755",
    package_dir = "bin/",
)

pkg_tar(
    name = "libtorchtrt",
    srcs = [
        "//:LICENSE",
        "//bzl_def:BUILD",
        "//bzl_def:WORKSPACE",
    ],
    extension = "tar.gz",
    package_dir = "torch_tensorrt",
    deps = [
        ":include",
        ":include_core",
        ":lib",
    ] + select({
        ":windows": [],
        "//conditions:default": [":bin"],
    }),
)
