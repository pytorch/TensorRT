load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

exports_files([
    "CMakeLists.txt",
    "executorch/CMakeLists.txt",
])

# The ExecuTorch external repository is source-layout only. Keep this BUILD
# overlay small and build only the static core archive needed by the reference
# runner path.
filegroup(
    name = "executorch_sources",
    srcs = glob(
        ["executorch/**"],
        exclude = [".git/**"],
    ),
)

# EXECUTORCH_ENABLE_LOGGING is pinned rather than left to follow
# CMAKE_BUILD_TYPE. Its default is off in Release, which compiles the archive
# with ET_LOG_ENABLED=0, but that is a CMake compile definition and
# rules_foreign_cc harvests only the .a, so the Bazel targets that call into it
# keep runtime/platform/log.h's default of 1. logf is an inline function whose
# body is guarded by that macro, so the two halves disagree on its definition
# and every ET_LOG and ET_CHECK_MSG message reaches an empty vlogf. A runner
# built that way dies without saying why.
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
        "EXECUTORCH_ENABLE_LOGGING": "ON",
    },
    install = False,
    lib_source = ":executorch_sources",
    out_include_dir = "",
    out_static_libs = ["libexecutorch_core.a"],
    targets = ["executorch_core"],
)

# extension_cuda MUST resolve to exactly ONE shared library in the process so
# its `thread_local caller_stream_` has a single definition across every DSO
# (the TensorRT backend, the CUDA/AOTI backend, and the runner). A plain
# cc_library with srcs does not guarantee this: cc_binary/cc_test consumers
# default to static-linking their cc_library deps, which would embed a private
# copy of the thread-local and silently break the cross-backend caller-stream
# handshake.
#
# Following this repo's convention for shared libraries (see cpp/lib/BUILD),
# build the implementation once as a cc_binary(linkshared=True), re-import the
# .so via cc_import, and expose it through a srcs-less cc_library that also
# carries the headers (with include remap) and the cudart dep. Consumers depend
# on ":extension_cuda"; caller_stream.cpp is only ever linked into the one .so.

# Private implementation. It carries hdrs + include_prefix/strip_include_prefix
# so caller_stream.cpp can find its own <executorch/extension/cuda/...> header
# (the physical tree has an extra executorch/ dir; a cc_binary's srcs headers
# get no such remap). alwayslink=True guarantees the object -- and thus
# getCallerStream/CallerStreamGuard -- is pulled into the shared object below,
# even though that source-less cc_binary references none of its symbols.
cc_library(
    name = "extension_cuda_impl",
    srcs = ["executorch/extension/cuda/caller_stream.cpp"],
    hdrs = [
        "executorch/extension/cuda/caller_stream.h",
        "executorch/extension/cuda/export.h",
    ],
    local_defines = [
        "C10_USING_CUSTOM_GENERATED_MACROS",
        "EXECUTORCH_EXTENSION_CUDA_BUILDING",
    ],
    include_prefix = "executorch",
    strip_include_prefix = "executorch",
    alwayslink = True,
    linkstatic = True,
    visibility = ["//visibility:private"],
    deps = ["@cuda//:cudart"],
)

# The single shared object. The explicit -soname pins DT_SONAME to the bare
# filename so consumers get a clean DT_NEEDED=libextension_cuda.so that the
# packaged $ORIGIN/../lib RPATH resolves. alwayslink on the impl above ensures
# the caller-stream symbols are actually present despite the empty srcs.
cc_binary(
    name = "libextension_cuda.so",
    linkopts = ["-Wl,-soname,libextension_cuda.so"],
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [":extension_cuda_impl"],
)

# Import the shared object as a linkable target (native cc_import takes no deps).
cc_import(
    name = "extension_cuda_shared",
    shared_library = ":libextension_cuda.so",
    visibility = ["//visibility:private"],
)

# Public consumer-facing target: srcs-less, so it never static-links
# caller_stream.cpp. It provides the caller-stream headers (with the executorch/
# include remap), the imported single .so, and the transitive cudart dep, so
# every consumer dynamically links one shared thread-local.
cc_library(
    name = "extension_cuda",
    hdrs = [
        "executorch/extension/cuda/caller_stream.h",
        "executorch/extension/cuda/export.h",
    ],
    include_prefix = "executorch",
    strip_include_prefix = "executorch",
    deps = [
        ":extension_cuda_shared",
        "@cuda//:cudart",
    ],
)

# The CUDA DeviceAllocator that serves device planned buffers. In the pinned
# ExecuTorch release this source is compiled only into the CUDA/AOTI delegate,
# so a binary that delegates to TensorRT alone cannot get one without linking a
# delegate it never calls. Compile it on its own instead. It has no delegate
# dependencies: cuda_runtime.h, the caller-stream header already in
# libextension_cuda.so, and the ExecuTorch runtime headers.
#
# This target names a path that goes away when the allocator stops living under
# backends/cuda upstream, so the pin cannot move forward while it is still here.
cc_library(
    name = "executorch_cuda_allocator",
    srcs = ["executorch/backends/cuda/runtime/cuda_allocator.cpp"],
    hdrs = ["executorch/backends/cuda/runtime/cuda_allocator.h"],
    include_prefix = "executorch",
    strip_include_prefix = "executorch",
    deps = [
        ":executorch_headers",
        ":extension_cuda",
        "@cuda//:cudart",
    ],
)

cc_library(
    name = "executorch_headers",
    hdrs = glob(
        [
            "executorch/runtime/**/*.h",
            "executorch/runtime/backend/**/*.h",
            "executorch/extension/**/*.h",
        ],
        allow_empty = True,
    ),
    defines = [
        "C10_USING_CUSTOM_GENERATED_MACROS",
    ],
    include_prefix = "executorch",
    strip_include_prefix = "executorch",
    includes = [
        "executorch/runtime/core/portable_type/c10",
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
        ["executorch/extension/data_loader/file_data_loader.cpp"],
        allow_empty = True,
    ),
    deps = [
        ":executorch_core",
        ":executorch_headers",
    ],
)
