load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "source_files",
    srcs = [
        "CMakeLists.txt",
        "README.md",
        "kv_cache_decode_check.cpp",
        "main.cpp",
    ],
)

cc_binary(
    name = "example_executorch_runner",
    srcs = ["main.cpp"],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
    ],
)

cc_binary(
    name = "kv_cache_decode_check",
    srcs = ["kv_cache_decode_check.cpp"],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "@cuda//:cudart",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
    ],
)
