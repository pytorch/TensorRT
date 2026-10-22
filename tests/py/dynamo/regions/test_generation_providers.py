# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import _providers


@pytest.fixture
def proposal():
    return {
        "source": "def kernel(): pass",
        "kernel_name": "kernel",
        "constants": [{"name": "BLOCK", "value": 256}],
        "grid": [4],
        "num_warps": 4,
        "num_stages": None,
    }


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_cli_uses_stdin_and_existing_account(monkeypatch, tmp_path, proposal, provider):
    monkeypatch.setattr(_providers.shutil, "which", lambda name: f"/bin/{name}")
    prompt = "Generate a kernel; `this is data`, not a shell command."

    def run(argv, *, input, cwd, timeout_s):
        assert input == prompt
        assert prompt not in argv
        assert cwd == tmp_path
        assert timeout_s == 3
        assert argv[argv.index("--model") + 1] == "account-model"
        assert not any("dangerously" in item for item in argv)
        if provider == "claude":
            assert "--safe-mode" in argv
            assert "--bare" not in argv
            assert argv[argv.index("--tools") + 1] == ""
            assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
            assert (
                json.loads(argv[argv.index("--json-schema") + 1])
                == _providers.PROPOSAL_SCHEMA
            )
            stdout = json.dumps({"structured_output": proposal, "is_error": False})
        else:
            assert "--ignore-user-config" not in argv
            assert argv[argv.index("--sandbox") + 1] == "read-only"
            assert "features.shell_tool=false" in argv
            assert "features.hooks=false" in argv
            assert "mcp_servers={}" in argv
            assert argv[-1] == "-"
            schema = Path(argv[argv.index("--output-schema") + 1])
            assert json.loads(schema.read_text()) == _providers.PROPOSAL_SCHEMA
            Path(argv[argv.index("--output-last-message") + 1]).write_text(
                json.dumps(proposal)
            )
            stdout = "CLI progress can accompany the result"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(_providers, "run_process", run)
    assert (
        _providers.propose(
            provider=provider,
            model="account-model",
            prompt=prompt,
            workdir=tmp_path,
            timeout_s=3,
        )
        == proposal
    )


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        ({"result": "not JSON"}, ValueError),
        ({"result": "[]"}, ValueError),
        (
            {"structured_output": {}, "is_error": True, "result": "Login required"},
            _providers.ProviderError,
        ),
    ],
)
def test_claude_rejects_invalid_or_failed_response(
    monkeypatch, tmp_path, response, expected_error
):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps(response), ""
        ),
    )
    with pytest.raises(expected_error):
        _providers.propose(
            provider="claude", model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )


def test_claude_accepts_result_json(monkeypatch, tmp_path, proposal):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps({"result": json.dumps(proposal)}), ""
        ),
    )
    assert (
        _providers.propose(
            provider="claude", model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )
        == proposal
    )


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_missing_cli_explains_login(monkeypatch, tmp_path, provider):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: None)
    with pytest.raises(_providers.ProviderError, match="login"):
        _providers.propose(
            provider=provider, model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )


def test_cli_launch_failure_is_a_provider_error(monkeypatch, tmp_path):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/codex")

    def run(*args, **kwargs):
        raise OSError("The CLI executable disappeared")

    monkeypatch.setattr(_providers, "run_process", run)
    with pytest.raises(_providers.ProviderError, match="Unable to launch"):
        _providers.propose(
            provider="codex", model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )


def test_codex_does_not_reuse_stale_reply(monkeypatch, tmp_path):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/codex")
    (tmp_path / "reply.json").write_text('{"stale":true}')
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    with pytest.raises(RuntimeError, match="without writing"):
        _providers.propose(
            provider="codex", model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )


def test_failure_uses_stdout_and_redacts_credentials(monkeypatch, tmp_path):
    secret = "some-secret-token-value"
    monkeypatch.setenv("EXAMPLE_API_TOKEN", secret)
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, "x" * 3000 + f" auth failed: {secret}", ""
        ),
    )
    with pytest.raises(_providers.ProviderError, match="auth failed") as error:
        _providers.propose(
            provider="claude", model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )
    assert secret not in str(error.value)
    assert secret not in (tmp_path / "provider.log").read_text()
    assert "[redacted]" in str(error.value)
    assert len(str(error.value)) < 1600


def test_invalid_claude_json_preserves_bounded_redacted_diagnostics(
    monkeypatch, tmp_path
):
    secret = "provider-private-token"
    monkeypatch.setenv("PROVIDER_API_TOKEN", secret)
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    output = "x" * 70000 + f"malformed response {secret}\x1b[31m"
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, output, "diagnostic"
        ),
    )
    with pytest.raises(ValueError, match="JSON proposal"):
        _providers.propose(
            provider="claude", model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )
    log = (tmp_path / "provider.log").read_text()
    assert len(log.encode()) <= 65536
    assert "truncated" in log and "malformed response [redacted]" in log
    assert "diagnostic" in log and "STATUS: exit 0" in log
    assert secret not in log and "\x1b" not in log


@pytest.mark.parametrize("provider", ["claude", "custom_provider:author"])
def test_provider_timeout_preserves_partial_output_without_credentials(
    monkeypatch, tmp_path, provider
):
    secret = "timeout-private-token"
    monkeypatch.setenv("PROVIDER_API_TOKEN", secret)
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")

    def run(*args, **kwargs):
        raise _providers.ProcessTimeoutError(
            3, f"partial {secret}".encode(), b"still compiling"
        )

    monkeypatch.setattr(_providers, "run_process", run)
    with pytest.raises(TimeoutError, match="3s timeout"):
        _providers.propose(
            provider=provider, model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )
    log = (tmp_path / "provider.log").read_text()
    assert "partial [redacted]" in log and "still compiling" in log
    assert secret not in log and "3s timeout" in log


def test_real_subprocess_timeout_retains_already_captured_output(tmp_path):
    with pytest.raises(_providers.ProcessTimeoutError) as failure:
        _providers.run_process(
            [
                sys.executable,
                "-c",
                "import sys,time; print('partial', flush=True); print('diagnostic', file=sys.stderr, flush=True); time.sleep(60)",
            ],
            input=None,
            cwd=tmp_path,
            timeout_s=0.5,
        )
    assert "partial" in failure.value.stdout
    assert "diagnostic" in failure.value.stderr


def test_run_process_transmits_input_without_a_shell(tmp_path):
    prompt = "$(touch should-not-exist); `literal text`"
    result = _providers.run_process(
        [sys.executable, "-c", "import sys; print(sys.stdin.read())"],
        input=prompt,
        cwd=tmp_path,
        timeout_s=5,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == prompt
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups are required")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_timeout_kills_descendants_even_if_parent_has_exited(tmp_path, parent_exits):
    script = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        + ("" if parent_exits else "time.sleep(60)\n")
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _providers.run_process(
            [sys.executable, "-c", script], input=None, cwd=tmp_path, timeout_s=1
        )
    assert time.monotonic() - started < 5
    child_pid = int((tmp_path / "child.pid").read_text())
    status = Path(f"/proc/{child_pid}/stat")
    for _ in range(100):
        try:
            state = status.read_text().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("The timed-out process left a running descendant")


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_cli_uses_caller_schema_for_evidence_requests(monkeypatch, tmp_path, provider):
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["read", "propose"]}},
        "required": ["action"],
        "additionalProperties": False,
    }
    response = {"action": "read"}
    monkeypatch.setattr(_providers.shutil, "which", lambda name: f"/bin/{name}")

    def run(argv, **kwargs):
        assert "ADDITIONAL EVIDENCE" in kwargs["input"]
        if provider == "claude":
            assert json.loads(argv[argv.index("--json-schema") + 1]) == schema
            stdout = json.dumps({"structured_output": response})
        else:
            path = Path(argv[argv.index("--output-schema") + 1])
            assert json.loads(path.read_text()) == schema
            Path(argv[argv.index("--output-last-message") + 1]).write_text(
                json.dumps(response)
            )
            stdout = ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(_providers, "run_process", run)
    assert (
        _providers.propose(
            provider=provider,
            model=None,
            prompt="Read relevant evidence, then propose.",
            workdir=tmp_path,
            timeout_s=3,
            schema=schema,
            evidence={"resources": ["task"]},
        )
        == response
    )


@pytest.mark.parametrize("object_adapter", [False, True])
def test_custom_provider_contract_runs_through_bounded_process(
    monkeypatch, tmp_path, proposal, object_adapter
):
    module = types.ModuleType("kernel_provider_fixture")
    schema = {"type": "object", "properties": {"source": {"type": "string"}}}
    evidence = {"resources": ["task"]}

    def author(**kwargs):
        assert kwargs == {
            "model": "sdk-model",
            "prompt": "Generate source.",
            "workdir": tmp_path,
            "timeout_s": 3,
            "schema": schema,
            "evidence": evidence,
        }
        return proposal

    module.author = types.SimpleNamespace(propose=author) if object_adapter else author
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def run(argv, *, input, cwd, timeout_s):
        assert input is None
        assert cwd == tmp_path
        assert timeout_s == 3
        assert argv[:2] == [sys.executable, "-m"]
        request_path = Path(argv[argv.index("--request") + 1])
        response_path = Path(argv[argv.index("--response") + 1])
        _providers._run_adapter(request_path, response_path)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(_providers, "run_process", run)
    assert (
        _providers.propose(
            provider=f"{module.__name__}:author",
            model="sdk-model",
            prompt="Generate source.",
            workdir=tmp_path,
            timeout_s=3,
            schema=schema,
            evidence=evidence,
        )
        == proposal
    )


def test_custom_provider_does_not_reuse_stale_reply(monkeypatch, tmp_path):
    (tmp_path / "provider-response.json").write_text('{"stale":true}')
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    with pytest.raises(_providers.ProviderError, match="without a response"):
        _providers.propose(
            provider="custom_provider:author",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
        )


def test_custom_provider_failure_redacts_credentials(monkeypatch, tmp_path):
    secret = "secret-sdk-provider-token"
    monkeypatch.setenv("CUSTOM_API_TOKEN", secret)
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 1, "", secret),
    )
    with pytest.raises(_providers.ProviderError, match="redacted") as error:
        _providers.propose(
            provider="custom_provider:author",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
        )
    assert secret not in str(error.value)
    assert secret not in (tmp_path / "provider.log").read_text()


def test_custom_provider_imports_in_a_real_subprocess(monkeypatch, tmp_path):
    module = tmp_path / "kernel_sdk_fixture.py"
    module.write_text(
        "def author(*, model, prompt, workdir, timeout_s, schema, evidence):\n"
        "    return {'model': model, 'prompt': prompt, 'cwd': str(workdir), "
        "'schema': schema, 'evidence': evidence}\n"
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    )
    directory = tmp_path / "attempt"
    schema = {"type": "object"}
    response = _providers.propose(
        provider="kernel_sdk_fixture:author",
        model="fixture-model",
        prompt="literal $(data)",
        workdir=directory,
        timeout_s=30,
        schema=schema,
        evidence={"task": "snapshot"},
    )
    assert response == {
        "model": "fixture-model",
        "prompt": "literal $(data)",
        "cwd": str(directory),
        "schema": schema,
        "evidence": {"task": "snapshot"},
    }
    assert "STATUS: exit 0" in (directory / "provider.log").read_text()


@pytest.mark.parametrize("provider", ["unknown", "module:", ":author", "../pkg:author"])
def test_unknown_provider_explains_importable_adapter(provider, tmp_path):
    with pytest.raises(ValueError, match="module:attribute"):
        _providers.propose(
            provider=provider, model=None, prompt="test", workdir=tmp_path, timeout_s=3
        )
