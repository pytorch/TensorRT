load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

test_suite(
    name = "executorch_backend_tests",
    tests = [
        ":test_caller_stream",
        ":test_executorch_binding_names",
        ":test_executorch_blob_header",
        ":test_executorch_weight_streaming_budget",
        ":test_shared_scratch_backend",
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

# The long timeout is what makes the fork case's per-child deadline mean anything:
# it reaps 24 children at 30 s each, so the path where they wedge budgets 720 s
# against Bazel's 300 s default for a target that names neither size nor timeout.
# Two source files because one of them is how the target sees that scratch_pool()
# is one registry across translation units; see the second file's comment.
cc_test(
    name = "test_shared_scratch_pool",
    timeout = "long",
    srcs = [
        "test_shared_scratch_pool.cpp",
        "test_shared_scratch_pool_other_tu.cpp",
    ],
    deps = [
        "//cpp:tensorrt_executorch_shared_scratch_pool",
        "//cpp:tensorrt_executorch_shared_scratch_pool_reset",
        "@googletest//:gtest_main",
    ],
)

# exclusive because the memory comparison reads device-wide free memory, which
# any other GPU target running at the same time would move.
cc_test(
    name = "test_shared_scratch_backend",
    timeout = "long",
    srcs = ["test_shared_scratch_backend.cpp"],
    tags = ["exclusive"],
    target_compatible_with = select({
        "//cpp:linux_x86_64": [],
        "//cpp:sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "//cpp:tensorrt_executorch_blob_header",
        "//cpp:tensorrt_executorch_pooled_scratch_install",
        "//cpp:tensorrt_executorch_shared_scratch_pool",
        "//cpp:tensorrt_executorch_shared_scratch_pool_test_hooks",
        "@executorch//:executorch_core",
        "@executorch//:executorch_headers",
        "@executorch//:extension_cuda",
        "@googletest//:gtest_main",
    ] + select({
        "//cpp:linux_x86_64": [
            "@cuda//:cudart",
            "@tensorrt//:nvinfer",
        ],
        "//cpp:sbsa": [
            "@cuda//:cudart",
            "@tensorrt_sbsa//:nvinfer",
        ],
        "//conditions:default": [],
    }),
)
