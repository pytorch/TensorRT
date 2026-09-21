# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "trtorchexec",
    srcs = [
        "main.cpp",
    ],
    deps = [
        "//cpp:torch_tensorrt",
        "@libtorch",
        "@libtorch//:caffe2",
    ],
)
