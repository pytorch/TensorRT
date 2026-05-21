load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

exports_files([
    "CMakeLists.txt",
])

# The ExecuTorch external repository is source-layout only. Keep this BUILD
# overlay small and build only the static core archive needed by the reference
# runner path.
genrule(
    name = "executorch_core_archive",
    srcs = ["CMakeLists.txt"],
    outs = ["libexecutorch_core.a"],
    cmd = """
set -e
command -v cmake >/dev/null 2>&1 || {
  echo "cmake is required to build @executorch//:libexecutorch_core.a" >&2
  exit 1
}
src_dir="$$(dirname "$(location CMakeLists.txt)")"
build_dir="$(@D)/executorch_cmake"
rm -rf "$$build_dir"
cmake -S "$$src_dir" -B "$$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_TESTING=OFF \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON
max_jobs="$${MAX_JOBS:-$$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
cmake --build "$$build_dir" --target executorch_core -j "$$max_jobs"
core_archive="$$(find "$$build_dir" -name libexecutorch_core.a -type f -print -quit)"
if [ -z "$$core_archive" ]; then
  echo "ExecuTorch build completed but libexecutorch_core.a was not found" >&2
  exit 1
fi
cp "$$core_archive" "$@"
""",
    local = True,
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
