load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

test_suite(
    name = "executorch_backend_tests",
    tests = [
        ":test_executorch_binding_names",
        ":test_executorch_blob_header",
    ],
)

cc_test(
    name = "test_executorch_binding_names",
    srcs = ["test_executorch_binding_names.cpp"],
    deps = [
        "//cpp:tensorrt_serialization",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_executorch_blob_header",
    srcs = ["test_executorch_blob_header.cpp"],
    deps = [
        "//cpp:tensorrt_executorch_blob_header",
        "@googletest//:gtest_main",
    ],
)
