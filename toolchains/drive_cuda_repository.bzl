"""Expose NVIDIA DRIVE CUDA files as the Bazel @cuda_driveos repository."""

def _require_file(root, relative, description):
    path = root.get_child(relative)
    if not path.exists:
        fail("{} is missing {} ({})".format(root, relative, description))

def _find_cudart(directory):
    if not directory or not directory.exists:
        return None

    # Prefer the development linker name, then the DRIVE major-version soname.
    for filename in ["libcudart.so", "libcudart.so.13"]:
        candidate = directory.get_child(filename)
        if candidate.exists:
            return candidate

    # Remain usable with later CUDA major versions without hardcoding their
    # complete patch-level soname. Sorting keeps repository evaluation stable.
    versioned = sorted([
        entry.basename
        for entry in directory.readdir()
        if entry.basename.startswith("libcudart.so.")
    ])
    return directory.get_child(versioned[0]) if versioned else None

def _drive_cuda_repository_impl(ctx):
    cuda_root_value = ctx.os.environ.get("TORCHTRT_DRIVE_CUDA_ROOT", "").strip()
    library_dir_value = ctx.os.environ.get("TORCHTRT_DRIVE_CUDA_LIB_DIR", "").strip()

    if not cuda_root_value:
        fail(
            "Set TORCHTRT_DRIVE_CUDA_ROOT to the CUDA root injected into the " +
            "build container by the NVIDIA runtime.",
        )

    cuda_root = ctx.path(cuda_root_value)
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

    library_directories = []
    if library_dir_value:
        library_directories.append(ctx.path(library_dir_value))
    library_directories.append(cuda_root.get_child("targets/aarch64-linux/lib"))

    cudart = None
    for library_directory in library_directories:
        cudart = _find_cudart(library_directory)
        if cudart:
            break

    if not cudart:
        fail(
            "Could not find libcudart.so or a versioned libcudart.so.* in: {}".format(
                ", ".join([str(directory) for directory in library_directories]),
            ),
        )

    ctx.symlink(cuda_root, "cuda")

    # Give the BUILD definition a stable linker path without modifying the
    # injected DRIVE toolkit or requiring an unversioned host-side symlink.
    ctx.file("lib/.keep", "")
    ctx.symlink(cudart, "lib/libcudart.so")
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

drive_cuda_repository = repository_rule(
    implementation = _drive_cuda_repository_impl,
    attrs = {
        "build_file": attr.label(mandatory = True),
    },
    configure = True,
    environ = [
        "TORCHTRT_DRIVE_CUDA_ROOT",
        "TORCHTRT_DRIVE_CUDA_LIB_DIR",
    ],
)
