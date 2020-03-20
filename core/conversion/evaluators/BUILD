package(default_visibility = ["//visibility:public"])

cc_library(
    name = "evaluators",
    hdrs = [
        "evaluators.h",
    ],
    srcs = [
        "NodeEvaluatorRegistry.cpp",
        "prim.cpp",
    ],
    deps = [
        "@libtorch//:libtorch",
        "//core/util:prelude",
    ],
    alwayslink = True,
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/evaluators/",
    srcs = ["evaluators.h"],
)
