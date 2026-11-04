# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Expose an installed Python package as a repository with a supplied BUILD file.

The repository name is independent of the Python import package name. The caller
supplies the interpreter executable used by its Python toolchain; this rule does
not search other Python environments or import the requested package.
"""

_FIND_PACKAGE = """
import importlib.util
import sys

package = sys.argv[1]
if not package.isidentifier():
    sys.exit(f"Expected a top-level Python import package name, got {package!r}")
spec = importlib.util.find_spec(package)
if spec is None:
    sys.exit(f"Package {package!r} is not installed in {sys.executable}")
locations = list(spec.submodule_search_locations or [])
if len(locations) != 1:
    sys.exit(f"Package {package!r} must have one package directory; found {locations!r}")
print(locations[0])
"""

def _new_whl_repository_impl(ctx):
    result = ctx.execute([
        ctx.path(ctx.attr.python),
        "-c",
        _FIND_PACKAGE,
        ctx.attr.package,
    ])
    if result.return_code != 0:
        fail("Cannot create @%s: %s" % (ctx.name, result.stderr.strip()))

    package_path = ctx.path(result.stdout.strip())
    if not package_path.exists or not package_path.is_dir:
        fail("Package %r did not resolve to a directory: %s" % (ctx.attr.package, package_path))

    # Preserve the installed package layout while supplying our own Bazel files.
    reserved = ["BUILD", "BUILD.bazel", "MODULE.bazel", "REPO.bazel", "WORKSPACE", "WORKSPACE.bazel"]
    for child in package_path.readdir():
        if child.basename not in reserved:
            ctx.symlink(child, child.basename)
    ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))

new_whl_repository = repository_rule(
    implementation = _new_whl_repository_impl,
    attrs = {
        "build_file": attr.label(mandatory = True, allow_single_file = True),
        "package": attr.string(mandatory = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
    },
    environ = ["PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONUSERBASE"],
    local = True,
)
