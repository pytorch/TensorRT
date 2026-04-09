load("@rules_cc//cc:defs.bzl", "cc_library")

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
