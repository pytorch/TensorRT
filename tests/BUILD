test_suite(
    name = "tests",
    tests = [
        "//tests/core/converters:test_converters",
        "//tests/modules:test_modules"
    ],
)

test_suite(
    name = "required_and_optional_tests",
    tests = [
        ":tests",
        "//tests/accuracy:test_accuracy"
    ]
)