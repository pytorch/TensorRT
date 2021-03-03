test_suite(
    name = "tests",
    tests = [
        "//tests/core:core_tests",
        "//tests/modules:module_tests"
        "//tests/api:test_apis"
    ],
)

test_suite(
    name = "required_and_optional_tests",
    tests = [
        ":tests",
        "//tests/accuracy:accuracy_tests"
    ]
)

test_suite(
   name = "aarch64_tests",
   tests = [
        "//tests/core:core_tests",
        "//tests/modules:aarch64_module_tests"
   ],
)

test_suite(
   name = "required_and_optional_aarch64_tests",
   tests = [
        ":aarch64_tests",
        "//tests/accuracy:aarch64_accuracy_tests"
   ]
)

test_suite(
    name = "python_api_tests",
    tests = [
        "//tests/py:test_api",
        "//tests/py:test_to_backend_api"
    ]
)
