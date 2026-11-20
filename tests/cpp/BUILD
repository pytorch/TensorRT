# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "test_runtime_api",
    srcs = ["test_runtime_api.cpp"],
    deps = [
        "//core/runtime",
        "//core/util/logging",
        "//cpp:torch_tensorrt",
        "@googletest//:gtest_main",
    ],
)

test_suite(
    name = "api_tests",
    tests = [
        ":test_runtime_api",
        "//tests/cpp/executorch:executorch_backend_tests",
    ],
)
