load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

# This BUILD file is used both by:
#   - toolchains/local_executorch.bzl  (symlinks runtime/, extension/, cmake-out/
#     into the synthetic repo root)
#   - new_local_repository(path = "${EXECUTORCH_PATH}")  (repo root IS the
#     executorch source tree)
#
# In both cases the repo root contains runtime/, extension/, and cmake-out/.
# include_prefix = "executorch" remaps runtime/... → <executorch/runtime/...>.

# Headers-only: provides include paths for compilation but does NOT link any
# static ExecuTorch library.  ExecuTorch runtime symbols (register_backend,
# find_backend, FreeableBuffer, …) are left undefined and resolved at dlopen()
# time from libqnn_executorch_backend.so, which _portable_lib.so loads with
# RTLD_GLOBAL.  This ensures all backends share the same registry instance.
cc_library(
    name = "executorch_headers",
    hdrs = glob([
        "runtime/**/*.h",
        "extension/**/*.h",
    ]),
    include_prefix = "executorch",
)

# ExecuTorch runtime shared library.  Carries 6 backend-registry symbols that
# libtrt_executorch_backend.so leaves undefined at link time.  Must be loaded
# with RTLD_GLOBAL before any TRT backend .so is dlopen'd so all backends share
# the same registry instance.
cc_import(
    name = "executorch_runtime",
    shared_library = "libqnn_executorch_backend.so",
)

# Full ExecuTorch C++ runtime (Program, Method, MethodMeta, runtime_init, …)
# built from the ExecuTorch cmake-out.  Link this into C++ binaries that call
# the ExecuTorch C++ API directly (e.g. trt_executor_runner).
cc_import(
    name = "executorch_core",
    static_library = "libexecutorch_core.a",
)

# FileDataLoader — lives in extension/data_loader/, compiled from the source
# tree that local_executorch.bzl symlinks into the synthetic repo root.
# Needs @libtorch for c10/util/safe_numerics.h.
cc_library(
    name = "executorch_file_data_loader",
    srcs = ["extension/data_loader/file_data_loader.cpp"],
    deps = [
        ":executorch_core",
        ":executorch_headers",
        "@libtorch",
    ],
)
