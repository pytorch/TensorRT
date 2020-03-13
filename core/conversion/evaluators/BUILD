package(default_visibility = ["//visibility:public"])

cc_library(
    name = "evaluators",
    hdrs = [
        "evaluators.h",
        "prim.cpp"
    ],
    srcs = [
        "NodeEvaluatorRegistry.cpp",
    ],
    deps = [
        "@libtorch//:libtorch",
        "//core/util:prelude",
    ]
)

load("@rules_pkg//:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "include",
    package_dir = "core/conversion/evaluators/",
    srcs = ["evaluators.h"],
)
