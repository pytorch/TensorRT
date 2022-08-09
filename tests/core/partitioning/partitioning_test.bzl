"""
Partitioning test macros
"""

load("@rules_cc//cc:defs.bzl", "cc_test")

def partitioning_test(name, visibility = None):
    """Macro to define a partitioning test

    Args:
        name: Name of test file
        visibility: Visibility of the test target
    """
    cc_test(
        name = name,
        srcs = [name + ".cpp"],
        visibility = visibility,
        deps = [
            "//tests/util",
            "@googletest//:gtest_main",
        ] + select({
            ":use_pre_cxx11_abi": ["@libtorch_pre_cxx11_abi//:libtorch"],
            "//conditions:default": ["@libtorch//:libtorch"],
        }),
        timeout = "short",
    )
