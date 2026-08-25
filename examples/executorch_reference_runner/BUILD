load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "source_files",
    srcs = [
        "CMakeLists.txt",
        "README.md",
        "kv_cache_decode_check.cpp",
        "load_model.py",
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
        "-Wl,-rpath,$$ORIGIN/../lib",
        "-ldl",
    ],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "//cpp:tensorrt_executorch_cuda_device_allocator",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
        "@executorch//:extension_cuda",
    ],
)

cc_binary(
    name = "kv_cache_decode_check",
    srcs = ["kv_cache_decode_check.cpp"],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "//cpp:tensorrt_executorch_cuda_device_allocator",
        "@cuda//:cudart",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
    ],
)
