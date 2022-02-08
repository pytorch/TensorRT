load("//tests/core/lowering:lowering_test.bzl", "lowering_test")

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

lowering_test(
    name = "test_linear_to_addmm",
)

cc_test(
  name = "test_module_fallback_passes",
  srcs = ["test_module_fallback_passes.cpp"],
  deps = [
      "//tests/util",
      "@googletest//:gtest_main",
  ] + select({
      ":use_pre_cxx11_abi":  ["@libtorch_pre_cxx11_abi//:libtorch"],
      "//conditions:default":  ["@libtorch//:libtorch"],
  }),
  data = [
      "//tests/modules:jit_models"
  ]
)

lowering_test(
    name = "test_conv1d_pass",
)

lowering_test(
    name = "test_remove_contiguous_pass",
)

lowering_test(
    name = "test_remove_dropout_pass",
)

lowering_test(
    name = "test_reduce_to_pass",
)

lowering_test(
    name = "test_reduce_gelu",
)

lowering_test(
    name = "test_remove_detach_pass",
)

lowering_test(
    name = "test_remove_unnecessary_casts",
)

lowering_test(
    name = "test_view_to_reshape_pass",
)

lowering_test(
    name = "test_operator_aliasing_pass",
)

lowering_test(
    name = "test_silu_to_sigmoid_multiplication",
)

lowering_test(
    name = "test_unpack_hardswish",
)

lowering_test(
    name = "test_unpack_reduce_ops",
)

test_suite(
    name = "lowering_tests",
    tests = [
        ":test_conv1d_pass",
        ":test_linear_to_addmm",
        ":test_module_fallback_passes",
        ":test_operator_aliasing_pass",
        ":test_remove_contiguous_pass",
        ":test_remove_detach_pass",
        ":test_view_to_reshape_pass",
        ":test_remove_dropout_pass",
        ":test_remove_unnecessary_casts",
        ":test_reduce_to_pass",
        ":test_reduce_gelu",
        ":test_unpack_hardswish",
        ":test_unpack_reduce_ops"
    ],
)
