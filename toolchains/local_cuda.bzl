"""Repository rule for selecting the CUDA toolkit used by a local build."""

def _local_cuda_impl(ctx):
    cuda_home = ctx.os.environ.get("CUDA_HOME", "").strip()
    if not cuda_home:
        cuda_home = ctx.os.environ.get("CUDA_PATH", "").strip()
    if not cuda_home:
        nvcc = ctx.which("nvcc")
        if nvcc:
            cuda_home = str(nvcc.dirname.dirname)
    if not cuda_home:
        fail(
            "Could not locate a CUDA toolkit. Set CUDA_HOME or CUDA_PATH to a " +
            "conventional toolkit root, or put nvcc on PATH.",
        )

    cuda_path = ctx.path(cuda_home)
    required_files = [
        "include/cuda_runtime_api.h",
        "lib64/libcudart.so",
    ]
    missing = [
        relative
        for relative in required_files
        if not cuda_path.get_child(relative).exists
    ]
    if missing:
        fail(
            "CUDA toolkit root '{}' is not conventional; missing {}. ".format(
                cuda_home,
                ", ".join(missing),
            ) +
            "On DRIVE OS, point CUDA_HOME at the merged toolkit view containing " +
            "include/, lib64/, bin/, and nvvm/.",
        )

    for subdirectory in ["include", "lib64", "bin", "nvvm", "targets"]:
        source = cuda_path.get_child(subdirectory)
        if source.exists:
            ctx.symlink(source, subdirectory)

    ctx.file("BUILD", ctx.read(Label("@//third_party/cuda:BUILD")))

local_cuda = repository_rule(
    implementation = _local_cuda_impl,
    configure = True,
    environ = ["CUDA_HOME", "CUDA_PATH", "PATH"],
)
