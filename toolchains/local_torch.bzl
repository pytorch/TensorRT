"""Repository rule that locates the locally installed PyTorch.

This avoids hardcoding torch's install path in MODULE.bazel, making the build
portable across machines regardless of Python version or venv location.

Discovery order:
  1. TORCH_PATH env var  — absolute path to the torch package directory
  2. VIRTUAL_ENV env var  — virtualenv / uv venv ($VIRTUAL_ENV/bin/python3)
  3. CONDA_PREFIX env var — conda environment ($CONDA_PREFIX/bin/python3)
  4. .venv/bin/python3 relative to the workspace root (uv / virtualenv default)
  5. venv/bin/python3 relative to the workspace root
  6. python3 / python on PATH
"""

def _find_python(ctx):
    """Return a path-like object for a Python that has torch importable, or None."""
    candidates = []

    # virtualenv / uv venv
    virtual_env = ctx.os.environ.get("VIRTUAL_ENV", "")
    if virtual_env:
        candidates.append(ctx.path(virtual_env + "/bin/python3"))
        candidates.append(ctx.path(virtual_env + "/bin/python"))

    # conda environment
    conda_prefix = ctx.os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(ctx.path(conda_prefix + "/bin/python3"))
        candidates.append(ctx.path(conda_prefix + "/bin/python"))

    # Common relative-to-workspace venv locations
    # ctx.workspace_root is the real workspace root (not the synthetic repo root)
    ws = ctx.workspace_root
    for rel in [
        ".venv/bin/python3",
        ".venv/bin/python",
        "venv/bin/python3",
        "venv/bin/python",
    ]:
        candidates.append(ws.get_child(rel.replace("/", ws.sep if hasattr(ws, "sep") else "/")))

    # System Python last
    for name in ["python3", "python"]:
        p = ctx.which(name)
        if p:
            candidates.append(p)

    for candidate in candidates:
        if candidate.exists:
            result = ctx.execute([candidate, "-c", "import torch"])
            if result.return_code == 0:
                return candidate
    return None

def _local_torch_impl(ctx):
    # 1. Env-var override (takes priority over auto-detection)
    torch_dir = ctx.os.environ.get("TORCH_PATH", "").strip()

    if not torch_dir:
        python = _find_python(ctx)
        if not python:
            fail(
                "Cannot locate a Python interpreter that has torch installed. " +
                "Either activate the project venv, or set TORCH_PATH to the " +
                "directory containing torch (e.g. /path/to/.venv/lib/pythonX.Y/site-packages/torch).",
            )
        result = ctx.execute(
            [python, "-c", "import torch, os; print(os.path.dirname(torch.__file__))"],
        )
        if result.return_code != 0:
            fail("Failed to get torch path from python: " + result.stderr)
        torch_dir = result.stdout.strip()

    if not torch_dir:
        fail("Torch path is empty. Set TORCH_PATH or ensure torch is installed in the active Python.")

    torch_path = ctx.path(torch_dir)

    # Symlink the subdirectories the BUILD file references into the synthetic repo
    for sub in ["include", "lib", "share"]:
        child = torch_path.get_child(sub)
        if child.exists:
            ctx.symlink(child, sub)

    ctx.file("BUILD", ctx.read(Label("@//third_party/libtorch:BUILD")))

local_torch = repository_rule(
    implementation = _local_torch_impl,
    environ = ["TORCH_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"],
)
