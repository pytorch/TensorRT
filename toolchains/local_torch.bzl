# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Repository rule that locates the locally installed PyTorch.

This avoids hardcoding torch's install path in MODULE.bazel, making the build
portable across machines regardless of Python version or venv location.

The `package` attribute selects the pip package to locate (default "torch");
e.g. package = "torch_tensorrt" with its own `build_file` for a prebuilt
torch_tensorrt install.

Discovery order:
  1. TORCH_PATH env var (package = "torch" only)  — absolute path to the torch package directory
  2. VIRTUAL_ENV env var  — virtualenv / uv venv ($VIRTUAL_ENV/bin/python3)
  3. CONDA_PREFIX env var — conda environment ($CONDA_PREFIX/bin/python3)
  4. .venv/bin/python3 relative to the workspace root (uv / virtualenv default)
  5. venv/bin/python3 relative to the workspace root
  6. python3 / python on PATH
"""

def _find_python(ctx):
    """Return a path-like object for a Python that has torch importable, or None."""

    # Use a Label anchor to get the real workspace root reliably under Bzlmod.
    ws = ctx.path(Label("//:MODULE.bazel")).dirname

    candidates = []

    # Workspace-relative venv first — the project's own .venv always wins.
    for rel in [
        ".venv/bin/python3",
        ".venv/bin/python",
        "venv/bin/python3",
        "venv/bin/python",
    ]:
        candidates.append(ctx.path(str(ws) + "/" + rel))

    # Active virtualenv / uv venv (only if it's not a different project's venv)
    virtual_env = ctx.os.environ.get("VIRTUAL_ENV", "")
    if virtual_env:
        candidates.append(ctx.path(virtual_env + "/bin/python3"))
        candidates.append(ctx.path(virtual_env + "/bin/python"))

    # conda environment
    conda_prefix = ctx.os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        candidates.append(ctx.path(conda_prefix + "/bin/python3"))
        candidates.append(ctx.path(conda_prefix + "/bin/python"))

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
    package = ctx.attr.package

    # 1. Env-var override (takes priority over auto-detection)
    torch_dir = ctx.os.environ.get("TORCH_PATH", "").strip() if package == "torch" else ""

    if not torch_dir:
        python = _find_python(ctx)
        if not python:
            fail(
                "Cannot locate a Python interpreter that has torch installed. " +
                "Either activate the project venv, or set TORCH_PATH to the " +
                "directory containing torch (e.g. /path/to/.venv/lib/pythonX.Y/site-packages/torch).",
            )
        result = ctx.execute(
            [python, "-c", "import importlib.util; print(importlib.util.find_spec('%s').submodule_search_locations[0])" % package],
        )
        if result.return_code != 0:
            fail("Failed to get " + package + " path from python: " + result.stderr)
        torch_dir = result.stdout.strip()

    if not torch_dir:
        fail(package + " path is empty. Set TORCH_PATH or ensure torch is installed in the active Python.")

    torch_path = ctx.path(torch_dir)

    # Validate that the installation has the expected C++ headers.
    c10_include = torch_path.get_child("include").get_child("c10")
    if package == "torch" and not c10_include.exists:
        fail(
            "torch at '" + torch_dir + "' is missing include/c10/ C++ headers. " +
            "Install a full PyTorch wheel (pip install torch) that includes dev headers, " +
            "or set TORCH_PATH to the correct directory.",
        )

    # Symlink the subdirectories the BUILD file references into the synthetic repo
    for sub in ["include", "lib", "share"]:
        child = torch_path.get_child(sub)
        if child.exists:
            ctx.symlink(child, sub)

    ctx.file("BUILD", ctx.read(ctx.attr.build_file))

local_torch = repository_rule(
    implementation = _local_torch_impl,
    attrs = {
        "build_file": attr.label(default = "@//third_party/libtorch:BUILD"),
        "package": attr.string(default = "torch"),
    },
    environ = ["TORCH_PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "PATH", "HOME"],
)
