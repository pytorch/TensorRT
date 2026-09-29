# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "source_files",
    srcs = [
        "CMakeLists.txt",
        "README.md",
        "kv_cache_decode_check.cpp",
        "load_model.py",
        "load_model_device_resident.py",
        "main.cpp",
    ],
)

cc_binary(
    name = "example_executorch_runner",
    srcs = ["main.cpp"],
    # -ldl for the runtime lookup of the CUDA driver's green-context entry points.
    # The driver itself is deliberately not a link dependency: the release build image
    # ships neither libcuda nor a stub.
    linkopts = [
        # The wheel installs this at torch_tensorrt/bin, so ../lib reaches Torch-TensorRT's own
        # libraries and ../../ reaches the other distributions in site-packages. Without the latter
        # two the binary cannot start at all: TensorRT and the CUDA runtime ship in their own
        # distributions, and nothing else adds them to the search path for a program that is not
        # launched through Python. The delegate library in the companion wheel already carries the
        # same two entries, which is why it loads and this did not.
        "-Wl,-rpath,$$ORIGIN/../lib",
        "-Wl,-rpath,$$ORIGIN/../../tensorrt_libs",
        "-Wl,-rpath,$$ORIGIN/../../nvidia/cu13/lib",
        "-ldl",
    ],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "//cpp:tensorrt_executorch_cuda_device_allocator",
        "//cpp:tensorrt_executorch_device_copy_kernels",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
        "@executorch//:extension_cuda",
    ],
)

# A convenience target for building this check out of a bazel checkout. Unlike
# the runner above, which //:bin packages into the release tar wherever this
# backend ships and so compiles on those packaging builds, nothing depends on
# this one: it ships as source in //:executorch_source_package and is compiled
# from the CMakeLists beside it, which is the path
# verify-executorch-reference-runner.sh drives. So this dependency list has to
# be kept in step with that one by hand -- an error here surfaces only when
# someone builds this target.
cc_binary(
    name = "kv_cache_decode_check",
    srcs = ["kv_cache_decode_check.cpp"],
    linkopts = [
        # Same reasoning as the runner above. This one carried no run path at all.
        "-Wl,-rpath,$$ORIGIN/../lib",
        "-Wl,-rpath,$$ORIGIN/../../tensorrt_libs",
        "-Wl,-rpath,$$ORIGIN/../../nvidia/cu13/lib",
    ],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "//cpp:tensorrt_executorch_cuda_device_allocator",
        "//cpp:tensorrt_executorch_device_copy_kernels",
        "@cuda//:cudart",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
        # Included and called directly (CallerStreamGuard), so declared directly
        # rather than reached through a transitive header, matching the sibling
        # runner above.
        "@executorch//:extension_cuda",
    ],
)
