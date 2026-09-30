"""Import the public native SDKs needed to compile the standalone delegate."""

_ENV = [
    "TORCHTRT_NATIVE_ROOT",
    "EXECUTORCH_ROOT",
    "EXECUTORCH_LIBRARY",
    "EXECUTORCH_EXTENSION_CUDA_LIBRARY",
    "TENSORRT_ROOT",
    "CUDA_ROOT",
]

def _path(ctx, name):
    value = ctx.os.environ.get(name, "")
    if not value:
        fail("Set --repo_env=%s=/path/to/native/dependency when building the Edge-LLM backend" % name)
    path = ctx.path(value)
    if not path.exists:
        fail("%s does not exist: %s" % (name, value))
    return path

def _link(ctx, source, target):
    if not source.exists:
        fail("Native SDK file is missing: %s" % source)
    ctx.symlink(source, target)

def _native_sdk_impl(ctx):
    torchtrt = _path(ctx, "TORCHTRT_NATIVE_ROOT")
    trt = _path(ctx, "TENSORRT_ROOT")
    cuda = _path(ctx, "CUDA_ROOT")
    _link(ctx, torchtrt.get_child("include"), "torchtrt_include")
    _link(ctx, torchtrt.get_child("lib/libexecutorch_trt_backend.a"), "lib/libexecutorch_trt_backend.a")
    _link(ctx, _path(ctx, "EXECUTORCH_ROOT"), "executorch")
    _link(ctx, _path(ctx, "EXECUTORCH_LIBRARY"), "lib/libexecutorch_core.a")
    _link(ctx, _path(ctx, "EXECUTORCH_EXTENSION_CUDA_LIBRARY"), "lib/libextension_cuda.so")
    _link(ctx, trt.get_child("include"), "tensorrt_include")
    _link(ctx, trt.get_child("lib/libnvinfer.so"), "lib/libnvinfer.so")
    _link(ctx, cuda.get_child("include"), "cuda_include")
    _link(ctx, cuda.get_child("lib64/libcudart.so"), "lib/libcudart.so")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_import(name = "torchtrt", static_library = "lib/libexecutorch_trt_backend.a")
cc_import(name = "executorch_core", static_library = "lib/libexecutorch_core.a")
cc_import(name = "extension_cuda", shared_library = "lib/libextension_cuda.so")
cc_import(name = "nvinfer", shared_library = "lib/libnvinfer.so")
cc_import(name = "cudart", shared_library = "lib/libcudart.so")
cc_library(
    name = "backend",
    hdrs = glob(["torchtrt_include/**/*.h", "executorch/**/*.h", "tensorrt_include/*.h", "cuda_include/**/*.h", "cuda_include/**/*.hpp"]),
    includes = ["torchtrt_include", ".", "executorch/runtime/core/portable_type/c10", "tensorrt_include", "cuda_include", "cuda_include/cccl"],
    defines = ["C10_USING_CUSTOM_GENERATED_MACROS", "ET_LOG_ENABLED=1"],
    deps = [":torchtrt", ":executorch_core", ":extension_cuda", ":nvinfer", ":cudart"],
    linkopts = ["-pthread", "-ldl"],
)
""")

native_sdk = repository_rule(
    implementation = _native_sdk_impl,
    environ = _ENV,
    local = True,
)
