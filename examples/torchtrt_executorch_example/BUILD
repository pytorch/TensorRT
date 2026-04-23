load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# C++ inference runner for .pte files compiled with Torch-TensorRT.
# Mirrors load_static_shape.py but drives the ExecuTorch C++ runtime API
# directly instead of Python pybindings.
#
# Build:
#   bazel build //examples/torchtrt_executorch_example:trt_executor_runner
#
# Run:
#   ./bazel-bin/examples/torchtrt_executorch_example/trt_executor_runner \
#       --model_path=model.pte [--num_runs=1]
cc_binary(
    name = "trt_executor_runner",
    srcs = ["executor_runner.cpp"],
    deps = [
        # Headers for ExecuTorch runtime/ and extension/ (<executorch/...>).
        "@executorch//:executorch_headers",
        # Static ExecuTorch C++ runtime: Program, Method, MethodMeta,
        # runtime_init, HierarchicalAllocator, MemoryManager, …
        "@executorch//:executorch_core",
        # FileDataLoader (extension/data_loader/file_data_loader.cpp).
        "@executorch//:executorch_file_data_loader",
        # Registers TensorRTBackend via static initialiser (alwayslink = True)
        # and links it directly into the application binary.
        "//core/runtime/executorch:tensorrt_executorch_backend",
    ],
)
