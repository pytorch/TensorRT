# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Structured proposals through agent CLIs or an explicit provider adapter."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

from ._util import strip_control, write_json


class ProviderError(RuntimeError):
    """The CLI cannot author a candidate; revising kernel code cannot fix it."""


class ProcessTimeoutError(TimeoutError):
    """Preserve output captured before the subprocess deadline expired."""

    def __init__(self, timeout_s: float, stdout: str, stderr: str) -> None:
        super().__init__(f"Process exceeded its {timeout_s:g}s timeout")
        self.stdout = stdout
        self.stderr = stderr


# Imports only the adapter, not torch_tensorrt, so an author turn does not pay
# for torch, TensorRT, or a CUDA context while it waits on a remote model.
# An adapter's TimeoutError exits with this status, which the controller treats
# like a slow CLI turn rather than an unusable provider.
_ADAPTER_TIMEOUT_STATUS = 124
_ADAPTER_BOOTSTRAP = f"""\
import importlib, json, sys
from pathlib import Path

module, attribute, request, response = sys.argv[1:]
adapter = getattr(importlib.import_module(module), attribute)
if isinstance(adapter, type):
    adapter = adapter()
author = getattr(adapter, "propose", adapter)
if not callable(author):
    raise TypeError("Provider adapter must be callable or expose propose()")
arguments = json.loads(Path(request).read_text(encoding="utf-8"))
arguments["workdir"] = Path(arguments["workdir"])
try:
    reply = author(**arguments)
except TimeoutError as error:
    print(f"{{type(error).__name__}}: {{error}}", file=sys.stderr)
    sys.exit({_ADAPTER_TIMEOUT_STATUS})
if not isinstance(reply, dict):
    raise TypeError("Provider adapter must return a JSON object")
Path(response).write_text(json.dumps(reply, allow_nan=False), encoding="utf-8")
"""

# codex has no single switch for a tool-free request: disable each tool
# surface and keep notification hooks and project documents out of the turn.
_CODEX_SETTINGS = (
    *(
        f"features.{feature}=false"
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
        )
    ),
    "features.skip_host_skill_discovery=true",
    'web_search="disabled"',
    "apps._default.enabled=false",
    "notify=[]",
    "project_doc_max_bytes=0",
    'approval_policy="never"',
)
_CODEX_LIST_TIMEOUT_S = 30.0
_CODEX_MCP_SERVERS: dict[str, tuple[str, ...]] = {}


def _decoded(stream: IO[bytes]) -> str:
    stream.seek(0)
    return stream.read().decode("utf-8", errors="replace")


def run_process(
    argv: Sequence[str],
    *,
    input: str | None,
    cwd: Path,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded subprocess, then kill its whole process group.

    Output goes to files rather than pipes: a descendant that inherits them
    cannot hold a finished process open until the deadline.
    """
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Process timeout must be finite and positive")
    with (
        tempfile.TemporaryFile() as stdin,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        stdin.write((input or "").encode("utf-8", errors="replace"))
        stdin.seek(0)
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode: int | None = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            returncode = None
        finally:
            # Descendants must not outlive the call, even after a normal exit.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output, errors = _decoded(stdout), _decoded(stderr)
    if returncode is None:
        raise ProcessTimeoutError(timeout_s, output, errors)
    return subprocess.CompletedProcess(argv, returncode, output, errors)


# Environment variables whose values are credentials
_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASS(?:WORD|WD)?|AUTH|CREDENTIAL", re.I)
# Credential formats recognized wherever they appear
_SECRET_VALUE = re.compile(
    r"\b(?:sk-(?:ant-)?|hf_|gh[pousr]_|github_pat_|xox[abpr]-|glpat-|AKIA|ya29\.)"
    r"[A-Za-z0-9_.-]{6,}"
)


def _redact(message: str) -> str:
    # Longest first, so a secret that contains another is removed whole.
    secrets = sorted(
        (
            value
            for name, value in os.environ.items()
            # Values of six or seven characters count only if they are not all
            # letters, and shorter ones never: plain words such as "true" would
            # erase log text.
            if _SECRET_NAME.search(name)
            and (len(value) >= 8 or (len(value) >= 6 and not value.isalpha()))
        ),
        key=len,
        reverse=True,
    )
    for value in secrets:
        message = message.replace(value, "[redacted]")
    message = re.sub(r"(?i)(bearer\s+)\S+", r"\1[redacted]", message)
    message = _SECRET_VALUE.sub("[redacted]", message)
    return strip_control(message)


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
    argv: Sequence[str], *, input: str | None, cwd: Path, timeout_s: float, name: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = run_process(argv, input=input, cwd=cwd, timeout_s=timeout_s)
    except ProcessTimeoutError as error:
        _write_provider_log(cwd, error.stdout, error.stderr, str(error))
        raise
    except OSError as error:
        _write_provider_log(cwd, "", "", str(error))
        raise ProviderError(f"Unable to launch {name}") from None
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


def _adapter_parts(provider: str) -> tuple[str, str]:
    from torch_tensorrt.region._generation import split_import_path

    try:
        return split_import_path(provider)
    except ValueError:
        raise ValueError(
            "Kernel generation provider must be 'claude', 'codex', or module:attribute"
        ) from None


def propose(
    *,
    provider: str,
    model: str | None,
    prompt: str,
    workdir: Path,
    timeout_s: float,
    schema: dict[str, Any],
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Request one reply that follows ``schema`` within ``timeout_s``.

    ``claude`` and ``codex`` run the logged-in CLI without tools; the prompt
    already lists any evidence. ``module:attribute`` selects a trusted
    importable adapter: a callable, an object exposing ``propose``, or a class
    constructed without arguments, called with keyword arguments ``model``, ``prompt``, ``workdir``, ``timeout_s``,
    ``schema``, and ``evidence``, that returns a JSON object. The adapter runs
    in a subprocess killed after ``timeout_s``; the ``timeout_s`` it receives
    is slightly shorter, so its own timeout can report first. Raising
    ``TimeoutError`` lets the search continue with the next author round; any
    other exception ends it. Adapters may use an SDK instead of either CLI.
    """
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
    if provider == "claude":
        return _claude(binary, model, prompt, workdir, timeout_s, schema)
    return _codex(binary, model, prompt, workdir, timeout_s, schema)


def _claude(
    binary: str,
    model: str | None,
    prompt: str,
    workdir: Path,
    timeout_s: float,
    schema: dict[str, Any],
) -> dict[str, Any]:
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
    if model is not None:
        argv.extend(("--model", model))
    result = _run_provider(
        argv, input=prompt, cwd=workdir, timeout_s=timeout_s, name="the claude CLI"
    )
    # The result envelope explains failures better than the exit status does.
    try:
        envelope = json.loads(result.stdout)
    except ValueError:
        envelope = None
    if not isinstance(envelope, dict):
        if result.returncode:
            detail = _diagnostic(result.stderr or result.stdout)
            raise ProviderError(
                f"claude exited with status {result.returncode}: {detail}"
            )
        raise ValueError("claude did not return a JSON proposal")
    subtype = envelope.get("subtype")
    if envelope.get("is_error") or subtype == "error_max_structured_output_retries":
        cause = envelope.get("result") or "; ".join(
            map(str, envelope.get("errors") or ())
        )
        detail = _diagnostic(str(cause)) or "no detail"
        if subtype == "error_max_structured_output_retries":
            # The model could not satisfy the schema this turn; a later one may.
            raise ValueError(f"claude could not produce a schema-valid reply: {detail}")
        reason = "" if subtype in (None, "success") else f" ({subtype})"
        raise ProviderError(f"claude could not generate a kernel{reason}: {detail}")
    proposal = envelope.get("structured_output")
    if isinstance(proposal, dict):
        return proposal
    return _json_object(envelope.get("result", ""), "claude")


def _codex_mcp_servers(
    binary: str, overrides: Sequence[str], cwd: Path, timeout_s: float
) -> tuple[str, ...]:
    """Name the MCP servers that ``codex exec`` would start with ``overrides``.

    codex merges ``-c`` tables into the user's configuration key by key, so
    ``mcp_servers={}`` changes nothing; each server must be disabled by name.
    Disabling a name that is not configured is a configuration error, so the
    servers are listed under the same overrides. The listing is cached for
    the life of the process.
    """
    if binary not in _CODEX_MCP_SERVERS:
        argv = [binary, "mcp", "list", "--json", *overrides]
        try:
            result = run_process(argv, input=None, cwd=cwd, timeout_s=timeout_s)
            if result.returncode:
                # stdout may list server environments; report only stderr.
                raise ValueError(
                    _diagnostic(result.stderr) or f"exit status {result.returncode}"
                )
            servers = tuple(entry["name"] for entry in json.loads(result.stdout))
            if not all(isinstance(name, str) for name in servers):
                raise TypeError("server names must be strings")
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ProviderError(
                "Unable to list codex MCP servers to disable them for kernel "
                f"authoring ({error}). Make `codex mcp list --json` succeed, or "
                "use another provider."
            ) from None
        _CODEX_MCP_SERVERS[binary] = servers
    return _CODEX_MCP_SERVERS[binary]


def _codex(
    binary: str,
    model: str | None,
    prompt: str,
    workdir: Path,
    timeout_s: float,
    schema: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    overrides = [
        argument for setting in _CODEX_SETTINGS for argument in ("-c", setting)
    ]
    servers = _codex_mcp_servers(
        binary, overrides, workdir, min(_CODEX_LIST_TIMEOUT_S, timeout_s / 2)
    )
    if servers:
        # A dotted -c path would split names such as "docs.search"; quoted keys
        # of one inline table do not. JSON strings are valid TOML strings.
        disabled = ", ".join(
            f"{json.dumps(name)}={{enabled=false}}" for name in servers
        )
        overrides.extend(("-c", f"mcp_servers={{{disabled}}}"))
    schema_path = workdir / "proposal.schema.json"
    reply_path = workdir / "reply.json"
    write_json(schema_path, schema)
    reply_path.unlink(missing_ok=True)
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
        *overrides,
    ]
    if model is not None:
        argv.extend(("--model", model))
    argv.append("-")
    result = _run_provider(
        argv,
        input=prompt,
        cwd=workdir,
        timeout_s=timeout_s - (time.monotonic() - started),
        name="the codex CLI",
    )
    if result.returncode:
        detail = _diagnostic(result.stderr or result.stdout)
        raise ProviderError(f"codex exited with status {result.returncode}: {detail}")
    if not reply_path.is_file():
        raise RuntimeError("codex finished without writing a kernel proposal")
    return _json_object(reply_path.read_text(encoding="utf-8"), "codex")


def _custom_propose(
    *,
    provider: str,
    model: str | None,
    prompt: str,
    workdir: Path,
    timeout_s: float,
    schema: dict[str, Any],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    module, attribute = _adapter_parts(provider)
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    request_path = workdir / "provider-request.json"
    response_path = workdir / "provider-response.json"
    response_path.unlink(missing_ok=True)
    write_json(
        request_path,
        {
            "model": model,
            "prompt": prompt,
            "workdir": str(workdir),
            # Leave the adapter time to report its own timeout before the kill.
            "timeout_s": timeout_s - min(5.0, timeout_s / 10),
            "schema": schema,
            "evidence": evidence,
        },
    )
    argv = [
        sys.executable,
        "-P",
        "-c",
        _ADAPTER_BOOTSTRAP,
        module,
        attribute,
        str(request_path),
        str(response_path),
    ]
    result = _run_provider(
        argv,
        input=None,
        cwd=workdir,
        timeout_s=timeout_s,
        name="the custom provider adapter",
    )
    if result.returncode == _ADAPTER_TIMEOUT_STATUS:
        detail = _diagnostic(result.stderr or result.stdout)
        raise TimeoutError(f"Provider {provider!r} timed out: {detail}")
    if result.returncode:
        detail = _diagnostic(result.stderr or result.stdout)
        raise ProviderError(f"Provider {provider!r} failed: {detail}")
    if not response_path.is_file():
        raise ProviderError(f"Provider {provider!r} finished without a response")
    return _json_object(response_path.read_text(encoding="utf-8"), provider)
