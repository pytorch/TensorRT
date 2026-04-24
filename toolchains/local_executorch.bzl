"""Repository rule that locates ExecuTorch from a local source tree or Python installation.

Discovery order:
  1. EXECUTORCH_ROOT env var  — absolute path to the ExecuTorch source/package directory
  2. VIRTUAL_ENV env var      — virtualenv / uv venv ($VIRTUAL_ENV/bin/python3)
  3. CONDA_PREFIX env var     — conda environment ($CONDA_PREFIX/bin/python3)
  4. .venv/bin/python3 relative to the workspace root (uv / virtualenv default)
  5. venv/bin/python3 relative to the workspace root
  6. python3 / python on PATH
"""

def _find_python(ctx):
    """Return a path to a Python interpreter that has executorch importable, or None."""
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
            result = ctx.execute([candidate, "-c", "import executorch"])
            if result.return_code == 0:
                return candidate
    return None

def _local_executorch_impl(ctx):
    # 1. Env-var override (takes priority over auto-detection).
    #    setup.py sets this when building via pip's isolated build environment.
    et_dir_str = ctx.os.environ.get("EXECUTORCH_ROOT", "").strip()

    if not et_dir_str:
        python = _find_python(ctx)
        if not python:
            fail(
                "Cannot locate a Python interpreter that has executorch installed. " +
                "Either activate the project venv, or set EXECUTORCH_ROOT to the " +
                "executorch package directory (e.g. .venv/lib/pythonX.Y/site-packages/executorch).",
            )
        result = ctx.execute(
            [python, "-c", "import executorch, os; print(os.path.realpath(next(iter(executorch.__path__))))"],
        )
        if result.return_code != 0:
            fail("Failed to get executorch path from Python: " + result.stderr)
        et_dir_str = result.stdout.strip()

    if not et_dir_str:
        fail("EXECUTORCH_ROOT is empty. Set it or ensure executorch is installed in the active Python.")

    et_dir = ctx.path(et_dir_str)

    # Validate that the installation has headers.
    has_wheel_layout = et_dir.get_child("include/executorch").exists
    has_source_layout = et_dir.get_child("runtime").exists
    if not has_wheel_layout and not has_source_layout:
        fail(
            "executorch at '" + et_dir_str + "' is missing headers. " +
            "Expected include/executorch/ (wheel) or runtime/ (source). " +
            "Install a full executorch package or set EXECUTORCH_ROOT correctly.",
        )

    # Normalize to source layout so the static BUILD file always works.
    # Wheel layout:   include/executorch/runtime/*.h  →  symlink as  runtime/
    # Source layout:  runtime/*.h                     →  symlink as  runtime/
    if has_wheel_layout:
        for sub in ["runtime", "extension"]:
            child = et_dir.get_child("include/executorch/" + sub)
            if child.exists:
                ctx.symlink(child, sub)
    else:
        for sub in ["runtime", "extension"]:
            child = et_dir.get_child(sub)
            if child.exists:
                ctx.symlink(child, sub)

    # Symlink libqnn_executorch_backend.so (ExecuTorch runtime carrier).
    qnn_so_candidates = [
        "backends/qualcomm/libqnn_executorch_backend.so",
        "cmake-out/backends/qualcomm/libqnn_executorch_backend.so",
        "libqnn_executorch_backend.so",
        "lib/libqnn_executorch_backend.so",
        "lib64/libqnn_executorch_backend.so",
    ]
    for rel in qnn_so_candidates:
        candidate = et_dir.get_child(rel)
        if candidate.exists:
            ctx.symlink(candidate, "libqnn_executorch_backend.so")
            break

    # Symlink libexecutorch_core.a (full C++ runtime for trt_executor_runner).
    core_candidates = [
        "cmake-out/libexecutorch_core.a",
        "lib/libexecutorch_core.a",
        "lib64/libexecutorch_core.a",
        "libexecutorch_core.a",
    ]
    for rel in core_candidates:
        candidate = et_dir.get_child(rel)
        if candidate.exists:
            ctx.symlink(candidate, "libexecutorch_core.a")
            break

    ctx.file("BUILD", ctx.read(Label("@//third_party/executorch:BUILD")))

local_executorch = repository_rule(
    implementation = _local_executorch_impl,
    environ = [
        "EXECUTORCH_ROOT",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    ],
)
