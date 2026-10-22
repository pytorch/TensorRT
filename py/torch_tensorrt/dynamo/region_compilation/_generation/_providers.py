# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Structured kernel proposals through an existing Claude or Codex CLI login."""

import json
import math
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class ProviderError(RuntimeError):
    """The CLI cannot author a candidate; revising kernel code cannot fix it."""


PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source": {"type": "string"},
        "kernel_name": {"type": "string"},
        "constants": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": ["boolean", "number"]},
                },
                "required": ["name", "value"],
            },
        },
        "grid": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
            "maxItems": 3,
        },
        "num_warps": {"type": ["integer", "null"]},
        "num_stages": {"type": ["integer", "null"]},
    },
    "required": [
        "source",
        "kernel_name",
        "constants",
        "grid",
        "num_warps",
        "num_stages",
    ],
}


def run_process(
    argv: Sequence[str],
    *,
    input: Optional[str],
    cwd: Path,
    timeout_s: float,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded subprocess and clean up its descendants, including on timeout."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Process timeout must be finite and positive")
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Process exceeded its {timeout_s:g}s timeout") from None
        finally:
            # Descendants can retain pipes after the original process exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _diagnostic(message: str) -> str:
    for name, value in os.environ.items():
        if len(value) >= 8 and re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", message)
    message = re.sub(r"\b(?:sk-|sk-ant-)[A-Za-z0-9_-]+", "[redacted]", message)
    return message.strip()[-1500:]


def _json_object(raw: str, provider: str) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        raise ValueError(f"{provider} did not return a JSON proposal") from None
    if not isinstance(result, dict):
        raise ValueError(f"{provider} proposal must be a JSON object")
    return result


def propose(
    *,
    provider: str,
    model: Optional[str],
    prompt: str,
    workdir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    """Ask an authenticated CLI for source; execution belongs to the evaluator."""
    if provider not in ("claude", "codex"):
        raise ValueError(f"Unsupported kernel generation provider: {provider!r}")
    binary = shutil.which(provider)
    if binary is None:
        login = "claude auth login" if provider == "claude" else "codex login"
        raise ProviderError(f"Install the {provider} CLI and run `{login}` first")
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    reply_path = workdir / "reply.json"
    reply_path.unlink(missing_ok=True)
    if provider == "claude":
        argv = [
            binary,
            "-p",
            "--safe-mode",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(PROPOSAL_SCHEMA),
            "--tools",
            "",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--no-session-persistence",
            "--no-chrome",
            "--disable-slash-commands",
            "--permission-mode",
            "dontAsk",
        ]
    else:
        schema_path = workdir / "proposal.schema.json"
        schema_path.write_text(json.dumps(PROPOSAL_SCHEMA), encoding="utf-8")
        argv = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(reply_path),
        ]
        for feature in (
            "shell_tool",
            "unified_exec",
            "code_mode",
            "code_mode_host",
            "apps",
            "plugins",
            "remote_plugin",
            "hooks",
            "multi_agent",
            "multi_agent_v2",
            "browser_use",
            "browser_use_external",
            "computer_use",
            "image_generation",
            "memories",
            "workspace_dependencies",
            "request_permissions_tool",
            "skill_search",
            "skill_mcp_dependency_install",
        ):
            argv.extend(("-c", f"features.{feature}=false"))
        for setting in (
            "features.skip_host_skill_discovery=true",
            'web_search="disabled"',
            "mcp_servers={}",
            "apps._default.enabled=false",
            "notify=[]",
            "project_doc_max_bytes=0",
            'approval_policy="never"',
        ):
            argv.extend(("-c", setting))
    if model is not None:
        argv.extend(("--model", model))
    if provider == "codex":
        argv.append("-")
    try:
        result = run_process(argv, input=prompt, cwd=workdir, timeout_s=timeout_s)
    except OSError:
        raise ProviderError(f"Unable to launch the {provider} CLI") from None
    if result.returncode:
        detail = _diagnostic(result.stderr or result.stdout)
        raise ProviderError(
            f"{provider} exited with status {result.returncode}: {detail}"
        )
    if provider == "codex":
        if not reply_path.is_file():
            raise RuntimeError("codex finished without writing a kernel proposal")
        return _json_object(reply_path.read_text(encoding="utf-8"), provider)
    envelope = _json_object(result.stdout, provider)
    if envelope.get("is_error"):
        detail = _diagnostic(str(envelope.get("result", "Agent request failed")))
        raise ProviderError(f"claude could not generate a kernel: {detail}")
    proposal = envelope.get("structured_output")
    if isinstance(proposal, dict):
        return proposal
    return _json_object(envelope.get("result", ""), provider)
