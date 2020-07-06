load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include_core",
    package_dir = "include/trtorch",
    deps = [
        "//core:include",
        "//core/conversion:include",
        "//core/conversion/conversionctx:include",
        "//core/conversion/converters:include",
        "//core/conversion/converters/impl/plugins:include",
        "//core/conversion/evaluators:include",
        "//core/conversion/tensorcontainer:include",
        "//core/conversion/var:include",
        "//core/execution:include",
        "//core/lowering:include",
        "//core/lowering/passes:include",
        "//core/util:include",
        "//core/util/logging:include",
    ],
)

pkg_tar(
    name = "include",
    srcs = [
        "//cpp/api:api_headers",
    ],
    package_dir = "include/trtorch/",
)

pkg_tar(
    name = "lib",
    srcs = [
        "//cpp/api/lib:libtrtorch.so",
    ],
    mode = "0755",
    package_dir = "lib/",
)

pkg_tar(
    name = "bin",
    srcs = [
        "//cpp/trtorchc",
    ],
    mode = "0755",
    package_dir = "bin/",
)

pkg_tar(
    name = "libtrtorch",
    srcs = [
        "//:LICENSE",
    ],
    extension = "tar.gz",
    package_dir = "trtorch",
    deps = [
        ":bin",
        ":include",
        ":include_core",
        ":lib",
    ],
)
