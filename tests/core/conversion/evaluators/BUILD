load("//tests/core/conversion/evaluators:evaluator_test.bzl", "evaluator_test")

config_setting(
    name = "use_torch_whl",
    values = {
        "define": "torch_src=whl",
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
