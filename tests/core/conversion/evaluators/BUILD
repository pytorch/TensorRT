load("//tests/core/conversion/evaluators:evaluator_test.bzl", "evaluator_test")

config_setting(
    name = "use_torch_whl",
    flag_values = {
        "//toolchains/dep_src:torch": "whl",
    },
)

config_setting(
    name = "sbsa",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "default",
    },
)

config_setting(
    name = "jetpack",
    constraint_values = [
        "@platforms//cpu:aarch64",
    ],
    flag_values = {
        "//toolchains/dep_collection:compute_libs": "jetpack",
    },
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

evaluator_test(
    name = "test_prim_evaluators",
)

evaluator_test(
    name = "test_aten_evaluators",
)

test_suite(
    name = "evaluator_tests",
    tests = [
        ":test_aten_evaluators",
        ":test_prim_evaluators",
    ],
)
