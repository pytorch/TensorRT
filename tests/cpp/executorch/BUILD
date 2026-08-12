load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

test_suite(
    name = "executorch_backend_tests",
    tests = [
        ":test_executorch_binding_names",
        ":test_executorch_blob_header",
        ":test_optimization_profile_selection",
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

# Asserts against the profile-selection policy only, so unlike the runtime tests
# it needs neither a GPU nor a TensorRT engine. It does need TensorRT headers,
# hence the same platform gate the backend carries.
cc_test(
    name = "test_optimization_profile_selection",
    srcs = ["test_optimization_profile_selection.cpp"],
    target_compatible_with = select({
        "//cpp:linux_x86_64": [],
        "//cpp:sbsa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//cpp:tensorrt_executorch_optimization_profile_selection",
        "@googletest//:gtest_main",
    ],
)
