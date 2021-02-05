load("//tests/core/lowering:lowering_test.bzl", "lowering_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    }
)

lowering_test(
  name = "test_remove_contiguous_pass",
)

lowering_test(
  name = "test_remove_to",
)

lowering_test(
  name = "test_remove_detach_pass",
)

test_suite(
    name = "lowering_tests",
    tests = [
        ":test_remove_contiguous_pass",
        ":test_remove_to",
        ":test_remove_detach_pass"
    ]
)