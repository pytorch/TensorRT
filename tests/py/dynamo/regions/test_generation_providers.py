# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import subprocess
import sys
import time
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
    assert "[redacted]" in str(error.value)
    assert len(str(error.value)) < 1600


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
