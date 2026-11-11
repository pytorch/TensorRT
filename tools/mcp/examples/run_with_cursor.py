#!/usr/bin/env python3
"""Run a Torch-TensorRT MCP workflow with Cursor and report per-run token usage."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MCP_CONFIG = REPO_ROOT / ".cursor" / "mcp.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Cursor Local Agent optimization against the Torch-TensorRT "
            "MCP server and print the runtime-reported token usage."
        )
    )
    parser.add_argument(
        "task",
        help='optimization request, or "-" to read it from stdin',
    )
    parser.add_argument(
        "--model",
        default="auto",
        help='Cursor model ID (default: "auto")',
    )
    parser.add_argument(
        "--mcp-config",
        type=Path,
        default=DEFAULT_MCP_CONFIG,
        help="Cursor MCP configuration file",
    )
    parser.add_argument(
        "--mcp-server",
        default="torch-tensorrt",
        help='stdio MCP server name (default: "torch-tensorrt")',
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "isolated MCP output directory; defaults to "
            ".mcp-runs/<date>_<time>_<8-character-id>"
        ),
    )
    parser.add_argument(
        "--usage-json",
        type=Path,
        help="optionally write an additional status and token usage JSON file",
    )
    return parser


def _read_task(value: str) -> str:
    task = sys.stdin.read() if value == "-" else value
    task = task.strip()
    if not task:
        raise ValueError("optimization task must not be empty")
    return task


def _resolve_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _default_output_root() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    return (REPO_ROOT / ".mcp-runs" / f"{timestamp}_{run_id}").resolve()


def _prepare_mcp_server(
    mcp_config_path: Path,
    server_name: str,
    output_root: Path | None,
) -> tuple[dict[str, Any], Path]:
    """Clone the admin policy with a unique output root to avoid IDE lock conflicts."""
    mcp_config_path = mcp_config_path.expanduser().resolve()
    config = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    try:
        server = config["mcpServers"][server_name]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"MCP server {server_name!r} is not defined in {mcp_config_path}"
        ) from error

    if server.get("type", "stdio") != "stdio":
        raise ValueError(f"MCP server {server_name!r} must use stdio transport")
    command = server.get("command")
    raw_args = server.get("args", [])
    if not isinstance(command, str) or not command:
        raise ValueError(f"MCP server {server_name!r} has no command")
    if not isinstance(raw_args, list) or not all(
        isinstance(value, str) for value in raw_args
    ):
        raise ValueError(f"MCP server {server_name!r} has invalid args")

    args = list(raw_args)
    try:
        config_index = args.index("--config") + 1
        source_admin = _resolve_path(
            args[config_index],
            base=mcp_config_path.parent.parent,
        )
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"MCP server {server_name!r} must provide '--config <path>'"
        ) from error

    admin = json.loads(source_admin.read_text(encoding="utf-8"))
    if output_root is None:
        output_root = _default_output_root()
    output_root.mkdir(parents=True, exist_ok=False)
    admin["output_root"] = str(output_root)
    isolated_admin = output_root / "effective-admin-config.json"
    isolated_admin.write_text(
        json.dumps(admin, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args[config_index] = str(isolated_admin)

    prepared: dict[str, Any] = {
        "command": command,
        "args": args,
    }
    if server.get("env") is not None:
        prepared["env"] = server["env"]
    if server.get("cwd") is not None:
        prepared["cwd"] = server["cwd"]
    return prepared, output_root


def _usage_payload(
    run_id: str,
    status: str,
    usage: Any,
    error_code: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "error_code": error_code,
        "error_message": error_message,
        "usage": None,
    }
    if usage is not None:
        payload["usage"] = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_tokens": usage.total_tokens,
        }
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mcp_job_summaries(output_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record_path in (output_root / ".jobs").glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        request = record.get("request")
        source_job_id = (
            request.get("source_job_id") if isinstance(request, dict) else None
        )
        summaries.append(
            {
                "job_id": record.get("job_id"),
                "operation": record.get("operation"),
                "status": record.get("status"),
                "output_dir": record.get("output_dir"),
                "source_job_id": source_job_id,
                "created_at": record.get("created_at"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "failure": record.get("failure"),
            }
        )
    return sorted(summaries, key=lambda item: item.get("created_at") or "")


def _run_summary(
    *,
    task: str,
    model: str,
    agent_id: str | None,
    started_at: str,
    finished_at: str,
    duration_ms: int | None,
    payload: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    error = None
    if payload["error_code"] or payload["error_message"]:
        error = {
            "code": payload["error_code"],
            "message": payload["error_message"],
        }
    return {
        "schema_version": "0.1",
        "status": payload["status"],
        "created_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "task": task,
        "cursor": {
            "agent_id": agent_id,
            "run_id": payload["run_id"],
            "model": model,
        },
        "token_usage": payload["usage"],
        "error": error,
        "mcp_jobs": _mcp_job_summaries(output_root),
    }


def _print_usage(payload: dict[str, Any]) -> None:
    usage = payload["usage"]
    print("\n\nToken usage")
    print(f"Run ID: {payload['run_id']}")
    print(f"Status: {payload['status']}")
    if payload["error_code"] or payload["error_message"]:
        print(f"Error code: {payload['error_code'] or 'unknown'}")
        print(f"Error: {payload['error_message'] or 'no error message reported'}")
    if usage is None:
        print("No token usage was reported for this run.")
        return
    print(f"Input: {usage['input_tokens']:,}")
    print(f"Output: {usage['output_tokens']:,}")
    print(f"Cache read: {usage['cache_read_tokens']:,}")
    print(f"Cache write: {usage['cache_write_tokens']:,}")
    if usage["reasoning_tokens"] is not None:
        print(f"Reasoning: {usage['reasoning_tokens']:,}")
    print(f"Total: {usage['total_tokens']:,}")


def main() -> int:
    args = _parser().parse_args()
    try:
        task = _read_task(args.task)
        output_root = (
            args.output_root.expanduser().resolve() if args.output_root else None
        )
        mcp_server, output_root = _prepare_mcp_server(
            args.mcp_config,
            args.mcp_server,
            output_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 1

    api_key = os.environ.get("CURSOR_API_KEY")
    if not api_key:
        print(
            "CURSOR_API_KEY is required; create one in the Cursor dashboard "
            "and export it before running this command.",
            file=sys.stderr,
        )
        return 1

    try:
        from cursor_sdk import (
            Agent,
            AgentOptions,
            LocalAgentOptions,
            StdioMcpServerConfig,
        )
    except ImportError:
        print(
            "cursor-sdk is not installed; run "
            "'python -m pip install -r "
            "tools/mcp/examples/requirements.txt'.",
            file=sys.stderr,
        )
        return 1

    prompt = f"""\
Complete the Torch-TensorRT optimization request below.

For lowering-pass graph optimization, follow this bounded workflow:
1. Run baseline analyze, compile, validate, and benchmark jobs without a
   lowering_pass_candidate_id request field. Poll every job to a terminal state and
   retrieve every report. Save the baseline job IDs and benchmark protocol.
2. Inspect the baseline lowered and partitioned evidence. Create one unique
   32-character lowercase hexadecimal candidate ID, then generate the complete
   candidate source as
   {output_root / ".lowering_pass_candidates"}/lowering_pass_candidate_<lowering_pass_candidate_id>.py.
   Never overwrite an existing candidate file.
3. Generate at most one semantics-preserving transformation. The generated
   file must register an @_aten_lowering_pass decorator without an index, use
   the (gm, settings) signature, preserve graph I/O and node metadata, and call
   clean_up_graph_after_modifications after graph edits. Do not modify model
   factories.
4. Run candidate analyze and compile jobs with that top-level
   lowering_pass_candidate_id request field, preserving all baseline model
   parameters. Then run validate and benchmark from the candidate compile job.
   Keep the generated source unchanged until all candidate jobs finish.
   Use the same model, inputs, seed, compilation settings, sample, warmup,
   and iterations, and treat candidate compile evidence as authoritative.
5. Require candidate validation to pass. Compare
   baseline["compiled"]["wall_statistics"]["median"] with
   candidate["compiled"]["wall_statistics"]["median"], and calculate
   baseline_compiled_median / candidate_compiled_median. The report's
   speedup["wall"] compares eager PyTorch with one compiled artifact and is not
   the candidate-versus-baseline speedup.
6. Keep the generated candidate only when the lowered evidence changed as
   intended and compiled median latency clearly improved. Otherwise delete its
   generated source file. Baseline jobs must always omit
   lowering_pass_candidate_id.

Use the configured torch-tensorrt MCP execution tools. Report concrete job IDs,
settings, protocols, compiled latency, calculated baseline-relative speedup,
correctness, and failures. Do not invent results or claim success from coverage
alone.

User request:
{task}
"""

    server = StdioMcpServerConfig(**mcp_server)
    print(f"MCP artifacts: {output_root}")
    started_at = _utc_now()
    agent_id: str | None = None
    duration_ms: int | None = None
    sdk_error: Exception | None = None
    try:
        with Agent.create(
            AgentOptions(
                model=args.model,
                api_key=api_key,
                local=LocalAgentOptions(cwd=REPO_ROOT),
                mcp_servers={args.mcp_server: server},
                name="Torch-TensorRT optimization",
            )
        ) as agent:
            agent_id = agent.agent_id
            run = agent.send(prompt)
            streamed = False
            for text in run.iter_text():
                streamed = True
                print(text, end="", flush=True)
            result = run.wait()
            if not streamed and result.result:
                print(result.result, end="")
            payload = _usage_payload(
                run.id,
                result.status,
                result.usage,
                result.error_code,
                result.error_message,
            )
            duration_ms = result.duration_ms
    except Exception as error:
        sdk_error = error
        print(f"\nCursor SDK run failed: {error}", file=sys.stderr)
        payload = _usage_payload(
            "unavailable",
            "error",
            None,
            type(error).__name__,
            str(error),
        )

    _print_usage(payload)
    finished_at = _utc_now()
    summary = _run_summary(
        task=task,
        model=args.model,
        agent_id=agent_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        payload=payload,
        output_root=output_root,
    )
    run_json = output_root / "run.json"
    run_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Run summary: {run_json}")

    if args.usage_json:
        usage_json = args.usage_json.expanduser().resolve()
        usage_json.parent.mkdir(parents=True, exist_ok=True)
        usage_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Usage JSON: {usage_json}")

    if sdk_error is not None:
        return 1
    return 0 if payload["status"] == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
