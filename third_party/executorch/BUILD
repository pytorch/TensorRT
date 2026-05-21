load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

exports_files([
    "CMakeLists.txt",
])

# The ExecuTorch external repository is source-layout only. Keep this BUILD
# overlay small and build only the static core archive needed by the reference
# runner path.
filegroup(
    name = "executorch_sources",
    srcs = glob(
        ["**"],
        exclude = [".git/**"],
    ),
)

cmake(
    name = "executorch_core",
    cache_entries = {
        "BUILD_TESTING": "OFF",
        "CMAKE_ARCHIVE_OUTPUT_DIRECTORY": "$$INSTALLDIR/lib",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        "EXECUTORCH_BUILD_EXTENSION_DATA_LOADER": "ON",
        "EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR": "ON",
        "EXECUTORCH_BUILD_EXTENSION_MODULE": "ON",
        "EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP": "ON",
        "EXECUTORCH_BUILD_PYBIND": "OFF",
    },
    install = False,
    lib_source = ":executorch_sources",
    out_include_dir = "",
    out_static_libs = ["libexecutorch_core.a"],
    working_directory = "executorch",
    targets = ["executorch_core"],
)

cc_library(
    name = "executorch_headers",
    hdrs = glob(
        [
            "runtime/**/*.h",
            "runtime/backend/**/*.h",
            "extension/**/*.h",
        ],
        allow_empty = True,
    ),
    defines = [
        "C10_USING_CUSTOM_GENERATED_MACROS",
    ],
    include_prefix = "executorch",
    includes = [
        "runtime/core/portable_type/c10",
    ],
)

# ExecuTorch runtime carrier — provides register_backend and related symbols.
# Must be loaded with RTLD_GLOBAL before any backend .so is dlopen'd.
cc_import(
    name = "executorch_runtime",
    shared_library = "libqnn_executorch_backend.so",
)

# FileDataLoader compiled from source (present in source-tree installs).
cc_library(
    name = "executorch_file_data_loader",
    # buildifier: disable=constant-glob
    srcs = glob(
        ["extension/data_loader/file_data_loader.cpp"],
        allow_empty = True,
    ),
    deps = [
        ":executorch_core",
        ":executorch_headers",
    ],
)
