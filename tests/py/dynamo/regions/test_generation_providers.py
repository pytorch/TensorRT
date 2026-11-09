# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import _providers

from ._generation_utils import process_ended, requires_proc, wait_until_ended

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)

SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["propose", "stop"]}},
    "required": ["action"],
    "additionalProperties": False,
}

# Logs its arguments, answers `mcp list --json`, and writes an exec reply.
FAKE_CODEX = """\
import json, os, sys
from pathlib import Path

arguments = sys.argv[1:]
with open(os.environ["FAKE_CODEX_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(arguments) + "\\n")
if arguments[:3] == ["mcp", "list", "--json"]:
    print(os.environ["FAKE_CODEX_SERVERS"])
    sys.exit(int(os.environ.get("FAKE_CODEX_STATUS", "0")))
reply = Path(arguments[arguments.index("--output-last-message") + 1])
reply.write_text(json.dumps({"action": "stop"}), encoding="utf-8")
"""

ADAPTER = """\
import sys



def author(*, model, prompt, workdir, timeout_s, schema, evidence):
    return {
        "model": model,
        "prompt": prompt,
        "workdir": str(workdir),
        "timeout_s": timeout_s,
        "schema": schema,
        "evidence": evidence,
        "imported": [name for name in ("torch", "torch_tensorrt") if name in sys.modules],
    }


class Author:
    def propose(self, **kwargs):
        return author(**kwargs)


instance = Author()


def slow(**kwargs):
    raise TimeoutError("model request exceeded its deadline")


def broken(**kwargs):
    raise RuntimeError("model API rejected the request")
"""


@pytest.fixture(autouse=True)
def codex_without_mcp_servers(monkeypatch):
    """Tests that mock run_process see only the codex exec call."""
    monkeypatch.setattr(_providers, "_CODEX_MCP_SERVERS", {"/bin/codex": ()})


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


@pytest.fixture
def fake_codex(monkeypatch, tmp_path):
    directory = tmp_path / "bin"
    directory.mkdir()
    binary = directory / "codex"
    binary.write_text(f"#!{sys.executable}\n{FAKE_CODEX}")
    binary.chmod(0o755)
    log = tmp_path / "codex.log"
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    monkeypatch.setattr(_providers, "_CODEX_MCP_SERVERS", {})

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return calls


def _settings(argv):
    return [argv[index + 1] for index, item in enumerate(argv) if item == "-c"]


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_cli_uses_stdin_and_existing_account(monkeypatch, tmp_path, proposal, provider):
    monkeypatch.setattr(_providers.shutil, "which", lambda name: f"/bin/{name}")
    prompt = "Generate a kernel; `this is data`, not a shell command."

    def run(argv, *, input, cwd, timeout_s):
        assert input == prompt
        assert prompt not in argv
        assert cwd == tmp_path
        assert 0 < timeout_s <= 3
        assert argv[argv.index("--model") + 1] == "account-model"
        assert not any("dangerously" in item for item in argv)
        if provider == "claude":
            assert "--safe-mode" in argv
            assert "--bare" not in argv
            assert argv[argv.index("--tools") + 1] == ""
            assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
            assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
            stdout = json.dumps({"structured_output": proposal, "is_error": False})
        else:
            assert "--ignore-user-config" not in argv
            assert argv[argv.index("--sandbox") + 1] == "read-only"
            assert {"features.shell_tool=false", "features.hooks=false"} <= set(
                _settings(argv)
            )
            # No configured servers, so there is nothing to disable.
            assert not any(s.startswith("mcp_servers") for s in _settings(argv))
            assert argv[-1] == "-"
            schema = Path(argv[argv.index("--output-schema") + 1])
            assert json.loads(schema.read_text()) == SCHEMA
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
            schema=SCHEMA,
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
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize(
    ("envelope", "expected_error", "message"),
    [
        # The model could not satisfy the schema: a retryable authoring failure.
        (
            {
                "subtype": "error_max_structured_output_retries",
                "is_error": True,
                "errors": ["candidates[0].grid must be an array", "retry limit"],
            },
            ValueError,
            r"schema-valid reply: candidates\[0\]\.grid must be an array; retry limit",
        ),
        (
            {
                "subtype": "error_max_turns",
                "is_error": True,
                "errors": ["Reached the maximum number of turns"],
            },
            _providers.ProviderError,
            r"\(error_max_turns\): Reached the maximum number of turns",
        ),
        (
            {"subtype": "success", "is_error": True, "result": "Invalid API key"},
            _providers.ProviderError,
            "generate a kernel: Invalid API key",
        ),
        (
            {"subtype": "success", "is_error": False, "result": "Here is a kernel."},
            ValueError,
            "JSON proposal",
        ),
    ],
)
def test_claude_failures_are_read_from_the_result_envelope(
    monkeypatch, tmp_path, returncode, envelope, expected_error, message
):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    stdout = json.dumps({"type": "result", **envelope})
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], returncode, stdout, "Warning: settings file ignored"
        ),
    )
    with pytest.raises(expected_error, match=message) as error:
        _providers.propose(
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )
    assert "settings file ignored" not in str(error.value)


@pytest.mark.parametrize(
    "status, error",
    [
        (529, _providers.ProviderUnavailableError),
        (429, _providers.ProviderUnavailableError),
        (500, _providers.ProviderUnavailableError),
        (401, _providers.ProviderError),
    ],
)
def test_claude_api_statuses_are_retryable_only_when_transient(
    monkeypatch, tmp_path, status, error
):
    # Rate limits and server errors may clear up; a rejected credential will not.
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": status,
        "result": "API Error",
    }
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, json.dumps(envelope), ""
        ),
    )
    with pytest.raises(error):
        _providers.propose(
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
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
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )
        == proposal
    )


_STOP = {"structured_output": {"action": "stop"}}


@pytest.mark.parametrize(
    "envelope, cost",
    [
        ({**_STOP, "total_cost_usd": 0.0123}, 0.0123),
        ({**_STOP, "total_cost_usd": 0}, 0),
        # A turn that fails still costs what it reports.
        (
            {
                "subtype": "error_max_structured_output_retries",
                "is_error": True,
                "errors": ["retry limit"],
                "total_cost_usd": 0.5,
            },
            0.5,
        ),
        (_STOP, None),
        ({**_STOP, "total_cost_usd": None}, None),
        ({**_STOP, "total_cost_usd": "0.1"}, None),
        ({**_STOP, "total_cost_usd": True}, None),
        ({**_STOP, "total_cost_usd": -1.0}, None),
        ({**_STOP, "total_cost_usd": float("nan")}, None),
    ],
)
def test_claude_records_only_the_spend_it_reports(
    monkeypatch, tmp_path, envelope, cost
):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(
        _providers,
        "run_process",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, json.dumps({"type": "result", **envelope}), ""
        ),
    )
    with contextlib.suppress(_providers.ProtocolError):
        _providers.propose(
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )
    spend = tmp_path / _providers.SPEND_FILENAME
    if cost is None:
        assert not spend.exists()
    else:
        assert json.loads(spend.read_text()) == {"total_cost_usd": cost}


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_missing_cli_explains_login(monkeypatch, tmp_path, provider):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: None)
    with pytest.raises(_providers.ProviderError, match="login"):
        _providers.propose(
            provider=provider,
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )


def test_cli_launch_failure_is_a_provider_error(monkeypatch, tmp_path):
    monkeypatch.setattr(_providers.shutil, "which", lambda _: "/bin/codex")

    def run(*args, **kwargs):
        raise OSError("The CLI executable disappeared")

    monkeypatch.setattr(_providers, "run_process", run)
    with pytest.raises(_providers.ProviderError, match="Unable to launch"):
        _providers.propose(
            provider="codex",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
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
            provider="codex",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )


@pytest.mark.parametrize(
    ("servers", "disabled"),
    [
        ([], None),
        (
            [
                {"name": "github", "enabled": True},
                {"name": "docs.search", "enabled": False},
            ],
            {"github": {"enabled": False}, "docs.search": {"enabled": False}},
        ),
    ],
)
def test_codex_disables_each_configured_mcp_server(
    fake_codex, monkeypatch, tmp_path, servers, disabled
):
    monkeypatch.setenv("FAKE_CODEX_SERVERS", json.dumps(servers))
    for turn in ("first", "second"):
        reply = _providers.propose(
            provider="codex",
            model=None,
            prompt="test",
            workdir=tmp_path / turn,
            timeout_s=30,
            schema=SCHEMA,
        )
        assert reply == {"action": "stop"}
    listing, *turns = fake_codex()
    assert listing[:3] == ["mcp", "list", "--json"]
    # The listing runs once, under the same isolation as each turn.
    assert len(turns) == 2
    assert "features.plugins=false" in _settings(listing)
    for argv in turns:
        assert argv[0] == "exec"
        settings = _settings(argv)
        assert set(_settings(listing)) <= set(settings)
        overrides = [s for s in settings if s.startswith("mcp_servers")]
        if disabled is None:
            assert overrides == []
        else:
            (override,) = overrides
            assert tomllib.loads(override) == {"mcp_servers": disabled}


@pytest.mark.parametrize(
    ("output", "status"), [("", 2), ("not JSON", 0), ('[{"enabled": true}]', 0)]
)
def test_codex_fails_closed_when_mcp_servers_cannot_be_listed(
    fake_codex, monkeypatch, tmp_path, output, status
):
    monkeypatch.setenv("FAKE_CODEX_SERVERS", output)
    monkeypatch.setenv("FAKE_CODEX_STATUS", str(status))
    with pytest.raises(_providers.ProviderError, match="codex mcp list --json"):
        _providers.propose(
            provider="codex",
            model=None,
            prompt="test",
            workdir=tmp_path / "turn",
            timeout_s=30,
            schema=SCHEMA,
        )
    assert [argv[:3] for argv in fake_codex()] == [["mcp", "list", "--json"]]


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
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )
    assert secret not in str(error.value)
    assert secret not in (tmp_path / "provider.log").read_text()
    assert "[redacted]" in str(error.value)
    assert len(str(error.value)) < 1600


@pytest.mark.parametrize(
    "name, value",
    [
        ("HF_AUTH", "hf_abcdefghijkl"),
        ("SHORT_TOKEN", "abc123"),
        ("DB_PASSWD", "p4ssw0rd"),
        ("UNRELATED", "ghp_abcdefghijklmnop"),
    ],
)
def test_redaction_covers_auth_names_short_tokens_and_known_formats(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    assert value not in _providers._redact(f"failed with {value} here")


@pytest.mark.parametrize(
    "name, value",
    [
        ("DB_PASSWORD", "correcthorse"),
        ("SERVICE_TOKEN", "QwErTyUiOpAsDf"),
        ("API_KEY", "abcdefgh"),
        ("MY_SECRET", "Zebrafish"),
    ],
)
def test_redaction_covers_letter_only_secrets_of_eight_or_more_characters(
    monkeypatch, name, value
):
    monkeypatch.setenv(name, value)
    assert value not in _providers._redact(f"failed with {value} here")


def test_redaction_keeps_ordinary_words(monkeypatch):
    monkeypatch.setenv("DISABLE_AUTH", "true")
    monkeypatch.setenv("AUTH_MODE", "device")
    assert _providers._redact("true device user") == "true device user"


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
            provider="claude",
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
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
        raise _providers.ProcessTimeoutError(3, f"partial {secret}", "still compiling")

    monkeypatch.setattr(_providers, "run_process", run)
    with pytest.raises(TimeoutError, match="3s timeout"):
        _providers.propose(
            provider=provider,
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
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


@pytest.mark.parametrize("leader_exits", [False, True])
def test_descendants_never_outlive_the_process_call(tmp_path, leader_exits):
    script = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\n"
        "print('answer', flush=True)\n" + ("" if leader_exits else "time.sleep(60)\n")
    )
    argv = [sys.executable, "-c", script]
    started = time.monotonic()
    if leader_exits:
        # A descendant holding the leader's output must not delay its result.
        result = _providers.run_process(argv, input=None, cwd=tmp_path, timeout_s=30)
        assert result.returncode == 0
        assert result.stdout.strip() == "answer"
    else:
        with pytest.raises(_providers.ProcessTimeoutError) as failure:
            _providers.run_process(argv, input=None, cwd=tmp_path, timeout_s=1)
        assert failure.value.stdout.strip() == "answer"
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
        pytest.fail("A descendant kept running after the call returned")


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_cli_uses_caller_schema_for_evidence_requests(monkeypatch, tmp_path, provider):
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string", "enum": ["read", "propose"]}},
        "required": ["action"],
        "additionalProperties": False,
    }
    response = {"action": "read"}
    prompt = "Read relevant evidence, then propose."
    monkeypatch.setattr(_providers.shutil, "which", lambda name: f"/bin/{name}")

    def run(argv, **kwargs):
        # The prompt already lists the catalog; only adapters receive evidence.
        assert kwargs["input"] == prompt
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
            prompt=prompt,
            workdir=tmp_path,
            timeout_s=3,
            schema=schema,
            evidence={"catalog": [{"resource": "task.json"}]},
        )
        == response
    )


@pytest.mark.parametrize("attribute", ["author", "instance", "Author"])
def test_custom_provider_runs_in_a_light_bounded_subprocess(
    monkeypatch, tmp_path, attribute
):
    (tmp_path / "kernel_sdk_fixture.py").write_text(ADAPTER)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    )
    directory = tmp_path / "attempt"
    evidence = {"catalog": [{"resource": "task.json"}]}
    response = _providers.propose(
        provider=f"kernel_sdk_fixture:{attribute}",
        model="fixture-model",
        prompt="literal $(data)",
        workdir=directory,
        timeout_s=30,
        schema=SCHEMA,
        evidence=evidence,
    )
    assert response == {
        "model": "fixture-model",
        "prompt": "literal $(data)",
        "workdir": str(directory),
        # The adapter's own timeout can fire before the process is killed.
        "timeout_s": 27.0,
        "schema": SCHEMA,
        "evidence": evidence,
        "imported": [],
    }
    assert "STATUS: exit 0" in (directory / "provider.log").read_text()


@pytest.mark.parametrize(
    "attribute,error,message",
    [
        # A timeout only loses this author turn; the search continues.
        ("slow", TimeoutError, "exceeded its deadline"),
        ("broken", _providers.ProviderError, "rejected the request"),
    ],
)
def test_custom_provider_timeouts_are_retryable_and_other_failures_are_not(
    monkeypatch, tmp_path, attribute, error, message
):
    (tmp_path / "kernel_sdk_fixture.py").write_text(ADAPTER)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    )
    with pytest.raises(error, match=message) as raised:
        _providers.propose(
            provider=f"kernel_sdk_fixture:{attribute}",
            model=None,
            prompt="test",
            workdir=tmp_path / "attempt",
            timeout_s=30,
            schema=SCHEMA,
        )
    assert isinstance(raised.value, _providers.ProviderError) == (
        error is _providers.ProviderError
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
            schema=SCHEMA,
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
            schema=SCHEMA,
        )
    assert secret not in str(error.value)
    assert secret not in (tmp_path / "provider.log").read_text()


# Records its process id, then waits on a model that never answers.
_WAITING_ADAPTER = """\
import os, time
from pathlib import Path


def wait(*, workdir, **kwargs):
    Path(workdir, "adapter.pid").write_text(str(os.getpid()))
    time.sleep(120)
    return {}
"""

_CONTROLLER = """\
import sys
from pathlib import Path

from torch_tensorrt.dynamo.region_compilation._generation import _providers

_providers.propose(
    provider="waiting_adapter:wait",
    model=None,
    prompt="test",
    workdir=Path(sys.argv[1]),
    timeout_s=120,
    schema={"type": "object"},
)
"""


@requires_proc
def test_custom_provider_ends_soon_after_its_controller_is_killed(
    monkeypatch, tmp_path
):
    # SIGKILL skips run_process's cleanup, which would kill the adapter.
    (tmp_path / "waiting_adapter.py").write_text(_WAITING_ADAPTER)
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((str(tmp_path), os.environ.get("PYTHONPATH", "")))
    )
    record = tmp_path / "attempt" / "adapter.pid"
    controller = subprocess.Popen(
        [sys.executable, "-c", _CONTROLLER, str(record.parent)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 120
        while not (record.exists() and record.read_text()):
            assert controller.poll() is None, "the controller failed"
            assert time.monotonic() < deadline, "the adapter never started"
            time.sleep(0.05)
    finally:
        controller.kill()
        controller.wait()
    pid = int(record.read_text())
    try:
        assert wait_until_ended(pid, timeout_s=15), "an orphaned adapter kept running"
    finally:
        if not process_ended(pid):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.parametrize(
    "provider", ["unknown", "module:", ":author", "../pkg:author", "pkg:obj.attr"]
)
def test_unknown_provider_explains_importable_adapter(provider, tmp_path):
    with pytest.raises(ValueError, match="module:attribute"):
        _providers.propose(
            provider=provider,
            model=None,
            prompt="test",
            workdir=tmp_path,
            timeout_s=3,
            schema=SCHEMA,
        )
