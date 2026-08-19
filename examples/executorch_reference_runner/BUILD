load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "source_files",
    srcs = [
        "CMakeLists.txt",
        "README.md",
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
        "@executorch//:executorch_core",
        "@executorch//:executorch_file_data_loader",
        "@executorch//:extension_cuda",
    ],
)
