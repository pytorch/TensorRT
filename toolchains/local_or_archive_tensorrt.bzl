"""Select a local TensorRT SDK when requested, otherwise download an archive."""

def _has_any(ctx, root, candidates):
    return any([ctx.path(root).get_child(candidate).exists for candidate in candidates])

def _local_or_archive_tensorrt_impl(ctx):
    tensorrt_root = ctx.os.environ.get(ctx.attr.root_env, "").strip()
    if not tensorrt_root:
        ctx.download_and_extract(
            url = ctx.attr.urls,
            type = ctx.attr.archive_type,
            stripPrefix = ctx.attr.strip_prefix,
        )
        ctx.file("BUILD", ctx.read(ctx.attr.archive_build_file))
        return

    header_candidates = [
        "include/NvInfer.h",
        "include/aarch64-linux-gnu/NvInfer.h",
        "include/x86_64-linux-gnu/NvInfer.h",
    ]
    library_candidates = [
        "lib/libnvinfer.so",
        "lib/aarch64-linux-gnu/libnvinfer.so",
        "lib/x86_64-linux-gnu/libnvinfer.so",
    ]
    if not _has_any(ctx, tensorrt_root, header_candidates):
        fail(
            "{}='{}' does not contain a supported NvInfer.h location".format(
                ctx.attr.root_env,
                tensorrt_root,
            ),
        )
    if not _has_any(ctx, tensorrt_root, library_candidates):
        fail(
            "{}='{}' does not contain a supported libnvinfer.so location".format(
                ctx.attr.root_env,
                tensorrt_root,
            ),
        )

    root = ctx.path(tensorrt_root)
    for subdirectory in ["include", "lib", "bin"]:
        source = root.get_child(subdirectory)
        if source.exists:
            ctx.symlink(source, subdirectory)

    ctx.file("BUILD", ctx.read(ctx.attr.local_build_file))

local_or_archive_tensorrt = repository_rule(
    implementation = _local_or_archive_tensorrt_impl,
    attrs = {
        "archive_build_file": attr.label(mandatory = True),
        "archive_type": attr.string(mandatory = True),
        "local_build_file": attr.label(mandatory = True),
        "root_env": attr.string(default = "TORCHTRT_TENSORRT_ROOT"),
        "strip_prefix": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
    },
    configure = True,
    environ = ["TORCHTRT_TENSORRT_ROOT"],
)
