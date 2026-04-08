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

cc_import(
    name = "executorch_core_lib",
    static_library = "cmake-out/libexecutorch_core.a",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "executorch_core",
    hdrs = glob([
        "runtime/**/*.h",
        "extension/**/*.h",
    ]),
    include_prefix = "executorch",
    deps = [":executorch_core_lib"],
)
