test_suite(
    name = "tests",
    tests = [
        ":cpp_api_tests",
        "//tests/core:core_tests",
    ],
)

test_suite(
    name = "required_and_optional_tests",
    tests = [
        ":tests",
        "//tests/accuracy:accuracy_tests",
    ],
)

test_suite(
    name = "aarch64_tests",
    tests = [
        "//tests/core:core_tests",
        "//tests/cpp:aarch64_api_tests",
    ],
)

test_suite(
    name = "required_and_optional_aarch64_tests",
    tests = [
        ":aarch64_tests",
        "//tests/accuracy:aarch64_accuracy_tests",
    ],
)

test_suite(
    name = "cpp_api_tests",
    tests = [
        "//tests/cpp:api_tests",
    ],
)

test_suite(
    name = "python_api_tests",
    tests = [
        "//tests/py:test_api",
        "//tests/py:test_to_backend_api",
    ],
)

test_suite(
    name = "python_required_and_optional_tests",
    tests = [
        ":python_api_tests",
        "//tests/py:py_calibrator_tests",
    ],
)
