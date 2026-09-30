"""Optional local TensorRT-Edge-LLM adapter dependency for the standalone build.

Set EDGELLM_INCLUDE_DIR to the companion cpp include root and
EDGELLM_EXECUTORCH_LIBRARY to libedgellmExecutorch.so using --repo_env.
The dependency is fetched only when an Edge-LLM backend target is requested.
"""

def _local_edge_llm_impl(ctx):
    include_dir = ctx.os.environ.get("EDGELLM_INCLUDE_DIR", "")
    library = ctx.os.environ.get("EDGELLM_EXECUTORCH_LIBRARY", "")
    if not include_dir or not library:
        fail("Edge-LLM backend builds require --repo_env=EDGELLM_INCLUDE_DIR=/path/to/cpp and --repo_env=EDGELLM_EXECUTORCH_LIBRARY=/path/to/libedgellmExecutorch.so")
    if not ctx.path(include_dir + "/executorch/vitExecutorchAdapter.h").exists:
        fail("EDGELLM_INCLUDE_DIR must contain executorch/vitExecutorchAdapter.h: " + include_dir)
    if not ctx.path(library).exists:
        fail("EDGELLM_EXECUTORCH_LIBRARY does not exist: " + library)
    ctx.symlink(include_dir, "include")
    ctx.symlink(library, "lib/libedgellmExecutorch.so")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "executorch_adapter_shared",
    shared_library = "lib/libedgellmExecutorch.so",
)

cc_library(
    name = "executorch_adapter",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    includes = ["include"],
    deps = [":executorch_adapter_shared"],
)

filegroup(
    name = "executorch_adapter_library",
    srcs = ["lib/libedgellmExecutorch.so"],
)
""")

local_edge_llm = repository_rule(
    implementation = _local_edge_llm_impl,
    environ = ["EDGELLM_INCLUDE_DIR", "EDGELLM_EXECUTORCH_LIBRARY"],
    local = True,
)
