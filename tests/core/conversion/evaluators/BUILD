load("//tests/core/conversion/evaluators:evaluator_test.bzl", "evaluator_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
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
