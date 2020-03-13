def converter_test(name, visibility=None):
    native.cc_test(
        name = name,
        srcs = [name + ".cpp"],
        visibility = visibility,
        deps = [
            "//tests/util",
            "//core",
            "@libtorch//:libtorch",
            "@googletest//:gtest_main",
        ],
        timeout="short"
    )
