# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Structured proposals through agent CLIs or an explicit provider adapter."""

import argparse
import importlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ._backends import get_backend


class ProviderError(RuntimeError):
    """The CLI cannot author a candidate; revising kernel code cannot fix it."""


class ProcessTimeoutError(TimeoutError):
    """Preserve output captured before the subprocess deadline expired."""

    def __init__(
        self, timeout_s: float, stdout: str | bytes | None, stderr: str | bytes | None
    ) -> None:
        super().__init__(f"Process exceeded its {timeout_s:g}s timeout")
        self.stdout = (
            stdout.decode("utf-8", errors="replace")
            if isinstance(stdout, bytes)
            else stdout or ""
        )
        self.stderr = (
            stderr.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes)
            else stderr or ""
        )


# Backward-compatible default; each backend owns its proposal schema.
PROPOSAL_SCHEMA = get_backend("triton").proposal_schema


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
        except subprocess.TimeoutExpired as error:
            raise ProcessTimeoutError(timeout_s, error.output, error.stderr) from None
        finally:
            # Descendants can retain pipes after the original process exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _redact(message: str) -> str:
    for name, value in os.environ.items():
        if len(value) >= 8 and re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", message)
    message = re.sub(r"\b(?:sk-|sk-ant-)[A-Za-z0-9_-]+", "[redacted]", message)
    message = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", message)
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", message)


def _diagnostic(message: str) -> str:
    return _redact(message).strip()[-1500:]


def _write_provider_log(directory: Path, stdout: str, stderr: str, status: str) -> None:
    """Keep bounded diagnostics even when a response cannot be parsed as JSON."""
    text = _redact(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}\nSTATUS: {status}\n")
    encoded = text.encode("utf-8")
    if len(encoded) > 65536:
        prefix = "[earlier provider output truncated]\n"
        text = prefix + encoded[-(65536 - len(prefix)) :].decode(
            "utf-8", errors="ignore"
        )
    (directory / "provider.log").write_text(text, encoding="utf-8")


def _run_provider(
    argv: Sequence[str], *, input: Optional[str], cwd: Path, timeout_s: float
) -> subprocess.CompletedProcess[str]:
    try:
        result = run_process(argv, input=input, cwd=cwd, timeout_s=timeout_s)
    except (OSError, TimeoutError) as error:
        _write_provider_log(
            cwd, getattr(error, "stdout", ""), getattr(error, "stderr", ""), str(error)
        )
        raise
    _write_provider_log(cwd, result.stdout, result.stderr, f"exit {result.returncode}")
    return result


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
    schema: Optional[dict[str, Any]] = None,
    evidence: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Request a proposal; backend schemas and evidence remain caller-owned.

    ``module:attribute`` selects a trusted importable adapter: a callable or an
    object exposing ``propose``. It receives keyword arguments ``model``,
    ``prompt``, ``workdir``, ``timeout_s``, ``schema``, and ``evidence`` and returns
    a JSON object. Adapters run in a subprocess under the same time limit as
    the built-in CLIs. They may use an SDK without requiring either CLI.
    """
    schema = PROPOSAL_SCHEMA if schema is None else schema
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Provider response schema must describe a JSON object")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Provider timeout must be finite and positive")
    if provider not in ("claude", "codex"):
        return _custom_propose(
            provider=provider,
            model=model,
            prompt=prompt,
            workdir=workdir,
            timeout_s=timeout_s,
            schema=schema,
            evidence=evidence,
        )
    binary = shutil.which(provider)
    if binary is None:
        login = "claude auth login" if provider == "claude" else "codex login"
        raise ProviderError(f"Install the {provider} CLI and run `{login}` first")
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if evidence is not None:
        prompt += "\nADDITIONAL EVIDENCE:\n" + json.dumps(evidence)
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
            json.dumps(schema),
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
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
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
        result = _run_provider(argv, input=prompt, cwd=workdir, timeout_s=timeout_s)
    except TimeoutError:
        raise
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


def _adapter_parts(provider: str) -> tuple[str, str]:
    module, separator, attribute = provider.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module.split("."))
        or not all(part.isidentifier() for part in attribute.split("."))
    ):
        raise ValueError(
            "Kernel generation provider must be 'claude', 'codex', or module:attribute"
        )
    return module, attribute


def _custom_propose(
    *,
    provider: str,
    model: Optional[str],
    prompt: str,
    workdir: Path,
    timeout_s: float,
    schema: dict[str, Any],
    evidence: Optional[dict[str, Any]],
) -> dict[str, Any]:
    _adapter_parts(provider)
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    request_path = workdir / "provider-request.json"
    response_path = workdir / "provider-response.json"
    response_path.unlink(missing_ok=True)
    request_path.write_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "prompt": prompt,
                "workdir": str(workdir),
                "timeout_s": timeout_s,
                "schema": schema,
                "evidence": evidence,
            },
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    argv = [
        sys.executable,
        "-m",
        "torch_tensorrt.dynamo.region_compilation._generation._providers",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]
    try:
        result = _run_provider(argv, input=None, cwd=workdir, timeout_s=timeout_s)
    except TimeoutError:
        raise
    except OSError as error:
        raise ProviderError("Unable to launch the custom provider adapter") from error
    if result.returncode:
        detail = _diagnostic(result.stderr or result.stdout)
        raise ProviderError(f"Provider {provider!r} failed: {detail}")
    if not response_path.is_file():
        raise ProviderError(f"Provider {provider!r} finished without a response")
    return _json_object(response_path.read_text(encoding="utf-8"), provider)


def _run_adapter(request_path: Path, response_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    module, attribute = _adapter_parts(request.pop("provider"))
    adapter = importlib.import_module(module)
    for part in attribute.split("."):
        adapter = getattr(adapter, part)
    author = getattr(adapter, "propose", adapter)
    if not callable(author):
        raise TypeError("Provider adapter must be callable or expose propose()")
    request["workdir"] = Path(request["workdir"])
    response = author(**request)
    if not isinstance(response, dict):
        raise TypeError("Provider adapter must return a JSON object")
    response_path.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one explicit provider adapter")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args()
    _run_adapter(arguments.request, arguments.response)
