load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "libtorchtrt_plugins.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/plugins:torch_tensorrt_plugins",
    ],
)

cc_binary(
    name = "libtorchtrt_runtime.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/plugins:torch_tensorrt_plugins",
        "//core/runtime",
    ],
)

cc_binary(
    name = "libtorchtrt.so",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp:torch_tensorrt",
    ],
)

cc_binary(
    name = "torchtrt.dll",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//cpp:torch_tensorrt",
    ],
)

cc_binary(
    name = "torchtrt_runtime.dll",
    srcs = [],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/plugins:torch_tensorrt_plugins",
        "//core/runtime",
    ],
)

cc_binary(
    name = "libtrt_executorch_backend.so",
    srcs = [],
    # Allow undefined symbols: ExecuTorch runtime functions (register_backend,
    # find_backend, FreeableBuffer, …) are intentionally left unresolved at
    # link time.  They are resolved at dlopen() time from
    # libqnn_executorch_backend.so, which _portable_lib.so has already loaded
    # into the process with RTLD_GLOBAL.
    linkopts = ["-Wl,--allow-shlib-undefined"],
    linkshared = True,
    linkstatic = True,
    deps = [
        "//core/runtime/executorch:tensorrt_executorch_backend",
    ],
)

pkg_files(
    name = "lib_pkg_files",
    srcs = ["torchtrt.dll"],
    prefix = "lib/",
    visibility = ["//visibility:public"],
)
