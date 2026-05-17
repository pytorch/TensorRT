load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files([
    "libexecutorch_core.a",
])

# Headers are normalized to source layout (runtime/, extension/) by local_executorch.bzl
# regardless of whether the installation is a pip wheel or a source tree.
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
    include_prefix = "executorch",
    includes = [
        "runtime/core/portable_type/c10",
    ],
    defines = [
        "C10_USING_CUSTOM_GENERATED_MACROS",
    ],
)

# ExecuTorch runtime carrier — provides register_backend and related symbols.
# Must be loaded with RTLD_GLOBAL before any backend .so is dlopen'd.
cc_import(
    name = "executorch_runtime",
    shared_library = "libqnn_executorch_backend.so",
)

# Full ExecuTorch C++ runtime (Program, Method, MethodMeta, runtime_init, …).
# Linked into C++ binaries that use the ExecuTorch C++ API (e.g. trt_executor_runner).
cc_import(
    name = "executorch_core",
    static_library = "libexecutorch_core.a",
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
