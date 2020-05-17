config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

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
            "//conditions:default":  ["@libtorch//:libtorch"],
        }),
        timeout="short"
    )
