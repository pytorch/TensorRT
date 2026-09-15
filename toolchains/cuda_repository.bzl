"""Expose conventional or NVIDIA DRIVE CUDA files as the Bazel @cuda repository."""

def _require_file(root, relative, description):
    path = root.get_child(relative)
    if not path.exists:
        fail("{} is missing {} ({})".format(root, relative, description))

def _mirror_conventional_cuda(ctx):
    root = ctx.path(ctx.attr.conventional_path)
    if not root.exists:
        fail("Conventional CUDA toolkit does not exist: {}".format(root))

    # Match new_local_repository semantics without changing the conventional
    # CUDA layout expected by third_party/cuda/BUILD.
    for entry in root.readdir():
        if entry.basename not in ["BUILD", "BUILD.bazel"]:
            ctx.symlink(entry, entry.basename)
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.conventional_build_file))

def _expose_drive_cuda(ctx):
    cuda_root_value = ctx.os.environ.get("TORCHTRT_DRIVE_CUDA_ROOT", "").strip()
    library_dir_value = ctx.os.environ.get("TORCHTRT_DRIVE_CUDA_LIB_DIR", "").strip()

    if not cuda_root_value:
        fail(
            "Set TORCHTRT_DRIVE_CUDA_ROOT to the CUDA root injected into the " +
            "build container by the NVIDIA runtime.",
        )
    if not library_dir_value:
        fail(
            "Set TORCHTRT_DRIVE_CUDA_LIB_DIR to the visible SBSA CUDA target-library " +
            "directory in the build container.",
        )

    cuda_root = ctx.path(cuda_root_value)
    library_dir = ctx.path(library_dir_value)
    _require_file(
        cuda_root,
        "targets/aarch64-linux/include/cuda_runtime_api.h",
        "the DRIVE common CUDA headers",
    )
    _require_file(
        cuda_root,
        "thor/targets/aarch64-linux/include/curand_kernel.h",
        "the Thor-specific CUDA headers",
    )
    _require_file(cuda_root, "bin/nvcc", "the target CUDA compiler")
    _require_file(library_dir, "libcudart.so", "the CUDA runtime library")

    ctx.symlink(cuda_root, "cuda")
    ctx.symlink(library_dir, "lib")
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.drive_build_file))

def _cuda_repository_impl(ctx):
    target_platform = ctx.os.environ.get("TORCHTRT_TARGET_PLATFORM", "").strip().lower()
    if target_platform == "driveos":
        _expose_drive_cuda(ctx)
    else:
        _mirror_conventional_cuda(ctx)

cuda_repository = repository_rule(
    implementation = _cuda_repository_impl,
    attrs = {
        "conventional_build_file": attr.label(mandatory = True),
        "conventional_path": attr.string(mandatory = True),
        "drive_build_file": attr.label(mandatory = True),
    },
    configure = True,
    environ = [
        "TORCHTRT_TARGET_PLATFORM",
        "TORCHTRT_DRIVE_CUDA_ROOT",
        "TORCHTRT_DRIVE_CUDA_LIB_DIR",
    ],
)
