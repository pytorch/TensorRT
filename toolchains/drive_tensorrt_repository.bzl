"""Expose the local DRIVE TensorRT SDK as @tensorrt_driveos, without downloads."""

def _drive_tensorrt_repository_impl(ctx):
    root_value = ctx.os.environ.get("TORCHTRT_TENSORRT_ROOT", "").strip()
    if not root_value:
        fail("Set TORCHTRT_TENSORRT_ROOT to the target-compatible DRIVE TensorRT SDK root.")

    root = ctx.path(root_value)

    # Match the AArch64 layout consumed by third_party/tensorrt/local/BUILD.
    for relative in [
        "include/aarch64-linux-gnu/NvInfer.h",
        "lib/aarch64-linux-gnu/libnvinfer.so",
    ]:
        if not root.get_child(relative).exists:
            fail("TORCHTRT_TENSORRT_ROOT='{}' is missing {}".format(root_value, relative))

    for subdirectory in ["include", "lib", "bin"]:
        source = root.get_child(subdirectory)
        if source.exists:
            ctx.symlink(source, subdirectory)
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

drive_tensorrt_repository = repository_rule(
    implementation = _drive_tensorrt_repository_impl,
    attrs = {
        "build_file": attr.label(mandatory = True),
    },
    configure = True,
    environ = ["TORCHTRT_TENSORRT_ROOT"],
)
