"""NCCL detection for PyTorch builds."""

def _find_nccl_include(repository_ctx, torch_path):
    """Find nccl.h from the pip nvidia-nccl package co-installed with torch."""
    # pip's nvidia-nccl package installs nccl.h at <site-packages>/nvidia/nccl/include/
    candidate = torch_path + "/../nvidia/nccl/include"
    result = repository_ctx.execute(["test", "-f", candidate + "/nccl.h"])
    if result.return_code == 0:
        return candidate
    return None

def _torch_nccl_detect_impl(repository_ctx):
    """Detect if PyTorch was built with NCCL support."""

    # Skip detection on non-Linux (NCCL not available)
    os_name = repository_ctx.os.name.lower()
    nccl_include_dir = ""
    if "linux" not in os_name:
        has_nccl = False
    else:
        # Find libtorch path using the venv python if available, else system python3
        for python_bin in ["python3", "python"]:
            result = repository_ctx.execute([
                python_bin,
                "-c",
                "import torch; import os; print(os.path.dirname(torch.__file__))",
            ])
            if result.return_code == 0:
                break

        if result.return_code != 0:
            has_nccl = False
        else:
            torch_path = result.stdout.strip()
            lib_path = torch_path + "/lib/libtorch_cuda.so"

            # Check for ProcessGroupNCCL symbol
            result = repository_ctx.execute([
                "grep",
                "-q",
                "ProcessGroupNCCL",
                lib_path,
            ])
            has_nccl = (result.return_code == 0)

            if has_nccl:
                found = _find_nccl_include(repository_ctx, torch_path)
                if found:
                    nccl_include_dir = found
                else:
                    # Cannot find nccl.h — disable to avoid build errors
                    has_nccl = False

    if has_nccl and nccl_include_dir:
        # Copy nccl.h into the repository so it remains valid after the
        # uv build sandbox that provided it is torn down.
        repository_ctx.execute(["mkdir", "-p", "nccl_include"])
        repository_ctx.execute(["cp", nccl_include_dir + "/nccl.h", "nccl_include/nccl.h"])
        nccl_headers_target = """
cc_library(
    name = "nccl_headers",
    hdrs = ["nccl_include/nccl.h"],
    strip_include_prefix = "nccl_include",
)
"""
    else:
        nccl_headers_target = """
cc_library(
    name = "nccl_headers",
)
"""

    # Generate BUILD file with config_setting
    repository_ctx.file("BUILD", """
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

bool_flag(
    name = "use_nccl",
    build_setting_default = {has_nccl},
)

config_setting(
    name = "nccl_enabled",
    flag_values = {{":use_nccl": "True"}},
)
{nccl_headers_target}
""".format(has_nccl = has_nccl, nccl_headers_target = nccl_headers_target))

torch_nccl_detect = repository_rule(
    implementation = _torch_nccl_detect_impl,
    local = True,  # Re-run on each build to detect changes
)

def if_torch_nccl(if_true, if_false = []):
    """Returns if_true if PyTorch has NCCL, else if_false."""
    return select({
        "@torch_nccl//:nccl_enabled": if_true,
        "//conditions:default": if_false,
    })
