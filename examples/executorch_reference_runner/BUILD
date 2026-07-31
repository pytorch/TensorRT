load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "source_files",
    srcs = [
        "CMakeLists.txt",
        "README.md",
        "main.cpp",
    ],
)

cc_binary(
    name = "example_executorch_runner",
    srcs = ["main.cpp"],
    linkopts = ["-Wl,-rpath,$$ORIGIN/../lib"],
    deps = [
        "//cpp:tensorrt_executorch_backend",
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
        "@executorch//:extension_cuda",
    ],
)
