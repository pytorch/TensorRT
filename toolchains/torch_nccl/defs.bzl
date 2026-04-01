"""NCCL detection for PyTorch builds."""

def _torch_nccl_detect_impl(repository_ctx):
    """Detect if PyTorch was built with NCCL support."""

    # Skip detection on non-Linux (NCCL not available)
    os_name = repository_ctx.os.name.lower()
    if "linux" not in os_name:
        has_nccl = False
    else:
        # Find libtorch path
        result = repository_ctx.execute([
            "python3",
            "-c",
            "import torch; import os; print(os.path.dirname(torch.__file__))",
        ])

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

    # Generate BUILD file with config_setting
    repository_ctx.file("BUILD", """
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(default_visibility = ["//visibility:public"])

bool_flag(
    name = "use_nccl",
    build_setting_default = {has_nccl},
)

config_setting(
    name = "nccl_enabled",
    flag_values = {{":use_nccl": "True"}},
)
""".format(has_nccl = has_nccl))

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
