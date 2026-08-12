load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Decides what ships in libtorchtrt.tar.gz. CMakeLists.txt references every
# runner source, so a file missing here aborts the whole configure step for
# anyone building from the tarball, not just its own target.
filegroup(
    name = "source_files",
    srcs = [
        "CMakeLists.txt",
        "README.md",
        "main.cpp",
        "multi_profile_benchmark.cpp",
        "multi_profile_main.cpp",
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
