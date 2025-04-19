"""
Lowering test macros
"""

load("@rules_cc//cc:defs.bzl", "cc_test")

def lowering_test(name, visibility = None):
    """Macro to define a test for a lowering pass

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
            ":windows": ["@libtorch_win//:libtorch"],
            "//conditions:default": ["@libtorch//:libtorch"],
        }),
        timeout = "short",
    )
