# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Expose a local POSIX Python executable for the NGC toolchain and repositories."""

def _local_python_repository_impl(ctx):
    requested = ctx.attr.interpreter_path
    python = ctx.path(requested) if "/" in requested else ctx.which(requested)
    if not python or not python.exists:
        fail("Cannot find Python interpreter %r" % requested)

    # Invoke the original path: symlinking only the executable into this repository
    # would lose the pyvenv.cfg lookup for an interpreter from a virtualenv.
    quoted_python = "'" + str(python).replace("'", "'\"'\"'") + "'"
    ctx.file("python", "#!/bin/sh\nexec %s \"$@\"\n" % quoted_python, executable = True)
    ctx.file("BUILD.bazel", 'exports_files(["python"], visibility = ["//visibility:public"])\n')

local_python_repository = repository_rule(
    implementation = _local_python_repository_impl,
    attrs = {
        "interpreter_path": attr.string(mandatory = True),
    },
    environ = ["PATH"],
    local = True,
)
