"""Repository rule that locates the locally installed ExecuTorch source tree.

Discovery order:
  1. EXECUTORCH_PATH env var — absolute path to the executorch source root
                               (the directory containing runtime/, extension/)
  2. import executorch from the active Python interpreter — walks up from the
     package directory to find the source root that contains runtime/backend/
  3. VIRTUAL_ENV / CONDA_PREFIX / .venv / system python3

Only the header files under runtime/ and extension/ are used; no cmake build
of libexecutorch_core.a is required.  ExecuTorch runtime symbols are resolved
at dlopen() time from libqnn_executorch_backend.so (loaded by _portable_lib.so).
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
    # Note: cmake-out is no longer needed (executorch_core static lib was removed).
    for sub in ["runtime", "extension"]:
        child = et_dir.get_child(sub)
        if child.exists:
            ctx.symlink(child, sub)

    # Expose libqnn_executorch_backend.so as a cc_import target so that
    # libtrt_executorch_backend.so can resolve 6 ExecuTorch backend-registry
    # symbols at dlopen() time (register_backend, find_backend, vlogf, …).
    qnn_so = et_dir.get_child("backends/qualcomm/libqnn_executorch_backend.so")
    if qnn_so.exists:
        ctx.symlink(qnn_so, "libqnn_executorch_backend.so")

    # Expose libexecutorch_core.a so that C++ binaries (e.g. trt_executor_runner)
    # can link the full ExecuTorch runtime (Program::load, Method::execute,
    # runtime_init, MethodMeta, …) statically.
    core_a = et_dir.get_child("cmake-out/libexecutorch_core.a")
    if core_a.exists:
        ctx.symlink(core_a, "libexecutorch_core.a")

    ctx.file("BUILD", ctx.read(Label("@//third_party/executorch:BUILD")))

local_executorch = repository_rule(
    implementation = _local_executorch_impl,
    environ = ["EXECUTORCH_PATH", "VIRTUAL_ENV", "CONDA_PREFIX"],
)
