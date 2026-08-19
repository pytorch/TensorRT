load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

test_suite(
    name = "executorch_backend_tests",
    tests = [
        ":test_caller_stream",
        ":test_executorch_binding_names",
        ":test_executorch_blob_header",
        ":test_executorch_weight_streaming_budget",
        ":test_shared_scratch_pool",
    ],
)

cc_test(
    name = "test_caller_stream",
    srcs = ["test_caller_stream.cpp"],
    deps = [
        "@executorch//:extension_cuda",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_executorch_binding_names",
    srcs = ["test_executorch_binding_names.cpp"],
    deps = [
        "//core/runtime:tensorrt_binding_names",
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

cc_test(
    name = "test_executorch_weight_streaming_budget",
    srcs = ["test_executorch_weight_streaming_budget.cpp"],
    deps = [
        "//cpp:tensorrt_executorch_weight_streaming_budget",
        "@googletest//:gtest_main",
    ],
)

cc_test(
    name = "test_shared_scratch_pool",
    srcs = ["test_shared_scratch_pool.cpp"],
    deps = [
        "//cpp:tensorrt_executorch_shared_scratch_pool",
        "@googletest//:gtest_main",
    ],
)
