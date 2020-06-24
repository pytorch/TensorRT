
def converter_test(name, visibility=None):
    native.cc_test(
        name = name,
        srcs = [name + ".cpp"],
        visibility = visibility,
        deps = [
            "//tests/util",
            "//core",
            "@googletest//:gtest_main",
        ] + select({
            ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
            ":use_pre_cxx11_abi_aarch64":  ["@libtorch_pre_cxx11_abi_aarch64//:libtorch"],
            ":aarch64":  ["@libtorch_aarch64//:libtorch"],
            "//conditions:default":  ["@libtorch//:libtorch"],
        }),
        timeout="short"
    )
