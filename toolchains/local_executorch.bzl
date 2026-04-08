"""Repository rule that locates the locally installed ExecuTorch source tree.

Discovery order:
  1. EXECUTORCH_PATH env var — absolute path to the executorch source root
                               (the directory containing runtime/, extension/, cmake-out/)
  2. import executorch from the active Python interpreter — walks up from the
     package directory to find the source root that contains runtime/backend/
  3. VIRTUAL_ENV / CONDA_PREFIX / .venv / system python3

The synthetic repo is structured with a single `executorch/` symlink pointing
at the source root so that headers resolve as <executorch/runtime/...> with
`includes = ["."]` in the BUILD file.

Requires a cmake-out/ build inside the executorch source tree containing
libexecutorch_core.a.  Run cmake from the source root first:

    cmake -S . -B cmake-out <options>
    cmake --build cmake-out
"""

def _find_python(ctx):
    candidates = []

    virtual_env = ctx.os.environ.get("VIRTUAL_ENV", "")
    if virtual_env:
        candidates.append(ctx.path(virtual_env + "/bin/python3"))
        candidates.append(ctx.path(virtual_env + "/bin/python"))

    conda_prefix = ctx.os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(ctx.path(conda_prefix + "/bin/python3"))
        candidates.append(ctx.path(conda_prefix + "/bin/python"))

    ws = ctx.workspace_root
    for rel in [".venv/bin/python3", ".venv/bin/python", "venv/bin/python3", "venv/bin/python"]:
        candidates.append(ws.get_child(rel))

    for name in ["python3", "python"]:
        p = ctx.which(name)
        if p:
            candidates.append(p)

    for candidate in candidates:
        if candidate.exists:
            return candidate
    return None

def _find_executorch_source(ctx):
    """Return the path to the executorch source root, or None."""

    # 1. Env-var override
    et_path = ctx.os.environ.get("EXECUTORCH_PATH", "").strip()
    if et_path:
        p = ctx.path(et_path)
        if p.exists:
            return p
        fail("EXECUTORCH_PATH is set to '{}' but that directory does not exist.".format(et_path))

    # 2. Python import — walk up from the package to find the source root
    python = _find_python(ctx)
    if python:
        result = ctx.execute([
            python,
            "-c",
            "\n".join([
                "import executorch, os",
                "pkg = os.path.dirname(executorch.__file__)",
                "# Walk upward from the package looking for runtime/backend/",
                "for d in [pkg, os.path.join(pkg, '..'), os.path.join(pkg, '..', '..')]:",
                "    d = os.path.realpath(d)",
                "    if os.path.isdir(os.path.join(d, 'runtime', 'backend')):",
                "        print(d)",
                "        break",
            ]),
        ])
        if result.return_code == 0 and result.stdout.strip():
            p = ctx.path(result.stdout.strip())
            if p.exists:
                return p

    return None

def _local_executorch_impl(ctx):
    et_dir = _find_executorch_source(ctx)
    if et_dir == None:
        fail(
            "Cannot locate the ExecuTorch source tree. " +
            "Set EXECUTORCH_PATH to the directory that contains runtime/, " +
            "extension/, and cmake-out/ (e.g. export EXECUTORCH_PATH=/path/to/executorch). " +
            "Ensure that cmake-out/libexecutorch_core.a has been built.",
        )

    # Symlink the subdirectories referenced by the BUILD file into the synthetic
    # repo root, mirroring the new_local_repository(path=EXECUTORCH_PATH) layout.
    # include_prefix = "executorch" in the BUILD file handles the header remapping.
    for sub in ["runtime", "extension", "cmake-out"]:
        child = et_dir.get_child(sub)
        if child.exists:
            ctx.symlink(child, sub)

    ctx.file("BUILD", ctx.read(Label("@//third_party/executorch:BUILD")))

local_executorch = repository_rule(
    implementation = _local_executorch_impl,
    environ = ["EXECUTORCH_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"],
)
