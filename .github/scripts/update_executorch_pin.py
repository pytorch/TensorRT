#!/usr/bin/env python3
"""Propose the newest ExecuTorch wheel and its recorded source commit as one pin.

The update workflow opens a pull request. Pin checks and delegate build/test jobs
must pass before that proposal is usable.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSIONS_FILE = _REPO_ROOT / "dev_dep_versions.yml"
_PROVENANCE_LIMIT = 64 * 1024

# This allowlist prevents a release version from rewriting unrelated dependencies
# or content-addressed wheel URLs. The repository guard inventories sites separately.
_PIN_SITES = (
    ".github/workflows/build_linux.yml",
    ".github/workflows/executorch-test-linux.yml",
    "MODULE.bazel",
    "docker/MODULE.bazel.docker",
    "docker/MODULE.bazel.ngc",
    "justfile",
    "pyproject.toml",
    "py/torch-tensorrt-executorch-runtime/README.md",
    "py/torch-tensorrt-executorch-runtime/pyproject.toml",
    "toolchains/ci_workspaces/MODULE.bazel.tmpl",
    "examples/executorch_reference_runner/README.md",
)
_CLAUSE = r"(?:===|==|>=|<=|~=|!=|<|>)\s*[^\s\"'`,;()]+"
_MARKER_VALUE = r"""(?:[a-z_]+|"[^"\n]*"|'[^'\n]*')"""
_MARKER_ATOM = rf"(?:\([ \t]*)*{_MARKER_VALUE}[ \t]*(?:===|==|>=|<=|~=|!=|<|>|not[ \t]+in|in)[ \t]*{_MARKER_VALUE}(?:[ \t]*\))*"
_REQUIREMENT = re.compile(
    r"(?<![0-9A-Za-z._-])executorch(?:\[[A-Za-z0-9_., -]+\])?\s*"
    rf"(?P<constraints>@\s*[^\s\"'`]+|{_CLAUSE}(?:\s*,\s*{_CLAUSE})*)"
    rf"(?:\s*;\s*{_MARKER_ATOM}(?:\s+(?:and|or)\s+{_MARKER_ATOM})*)?",
    re.IGNORECASE,
)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as error:
        print(error.stderr or error.stdout or str(error), file=sys.stderr)
        raise


def read_pin(field: str) -> str:
    """Read a string scalar using the same YAML semantics as package metadata."""
    try:
        values = yaml.safe_load(_VERSIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SystemExit(f"cannot read {_VERSIONS_FILE}: {error}") from error
    value = values.get(field) if isinstance(values, dict) else None
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{field} is not a string pin in {_VERSIONS_FILE}")
    return value


def available_versions(index_args: list[str]) -> list[str]:
    """Read pip's version list from the index selected by CLI track/channel."""
    out = _run(
        [sys.executable, "-m", "pip", "index", "versions", "executorch", *index_args]
    )
    match = re.search(r"^\s*Available versions:\s*(.+)$", out, re.MULTILINE)
    if match is None:
        raise SystemExit("pip index versions printed no Available versions line")
    return [v.strip() for v in match.group(1).split(",") if v.strip()]


def pick_target(versions: list[str], track: str) -> str:
    """Select by PEP 440 order and omit the CUDA-specific local label."""
    parsed = []
    for raw in versions:
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        if track == "nightly" and version.dev is None:
            continue
        if track == "stable" and version.is_prerelease:
            continue
        parsed.append(version)
    if not parsed:
        raise SystemExit(f"no executorch version on the index matches track {track!r}")
    return max(parsed).public


def wheel_git_version(version: str, index_args: list[str]) -> str:
    """Read bounded wheel provenance without executing downloaded Python."""
    with tempfile.TemporaryDirectory() as tmp:
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--dest",
                tmp,
                f"executorch=={version}",
                *index_args,
            ]
        )
        wheels = list(Path(tmp).glob("executorch-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(
                f"expected one provenance wheel for executorch=={version}, got {len(wheels)}"
            )
        try:
            with zipfile.ZipFile(wheels[0]) as archive:
                member = archive.getinfo("executorch/version.py")
                if member.file_size > _PROVENANCE_LIMIT:
                    raise ValueError("version member exceeds 64 KiB")
                with archive.open(member) as stream:
                    body = stream.read(_PROVENANCE_LIMIT + 1)
                if len(body) > _PROVENANCE_LIMIT:
                    raise ValueError("version member exceeds 64 KiB")
                source = body.decode("utf-8")
        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
            zipfile.BadZipFile,
        ) as error:
            raise SystemExit(
                f"cannot read provenance for executorch=={version}: {error}"
            ) from error
    match = re.search(
        r"""^git_version[^=\n]*=\s*['"]([0-9a-f]{40})['"]""", source, re.MULTILINE
    )
    if match is None:
        raise SystemExit(f"executorch=={version} records no valid source provenance")
    return match.group(1)


def _upper_bound(version: str) -> str:
    parsed = Version(version)
    return f"{parsed.major}.{parsed.minor + 1}"


def _pin_site_paths() -> list[Path]:
    tracked = set(_run(["git", "-C", str(_REPO_ROOT), "ls-files"]).splitlines())
    missing = sorted(name for name in _PIN_SITES if name not in tracked)
    if missing:
        raise SystemExit(
            f"pin sites are not tracked: {missing}; update the writer and guard if a site moved"
        )
    return [*(_REPO_ROOT / name for name in _PIN_SITES), _VERSIONS_FILE]


def _rewrite_versions(text: str, version: str, commit: str) -> str:
    values = yaml.safe_load(text)
    node = yaml.compose(text)
    if not isinstance(node, yaml.MappingNode) or not isinstance(values, dict):
        raise ValueError("version source must be a YAML mapping")
    replacements = {"__executorch_version__": version, "__executorch_commit__": commit}
    edits = []
    for key, value in node.value:
        if key.value not in replacements:
            continue
        if not isinstance(value, yaml.ScalarNode) or value.style not in (
            None,
            "'",
            '"',
        ):
            raise ValueError(f"{key.value} must be a plain or quoted scalar")
        quote = value.style or ""
        edits.append(
            (
                value.start_mark.index,
                value.end_mark.index,
                quote + replacements[key.value] + quote,
            )
        )
    if len(edits) != 2:
        raise ValueError("version source must contain each pin exactly once")
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    if yaml.safe_load(text) != {**values, **replacements}:
        raise ValueError("version rewrite changed unrelated YAML fields")
    return text


def write_pins(new_version: str, new_commit: str) -> bool:
    """Preflight every input and proposed replacement before writing any pin site."""
    old_version = read_pin("__executorch_version__")
    old_commit = read_pin("__executorch_commit__")
    for version in (old_version, new_version):
        try:
            if Version(version).local:
                raise ValueError("the shared pin must not carry a CUDA local label")
        except (InvalidVersion, ValueError) as error:
            raise SystemExit(
                f"invalid public ExecuTorch pin {version!r}: {error}"
            ) from error
    for commit in (old_commit, new_commit):
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise SystemExit(f"invalid ExecuTorch source commit {commit!r}")

    exact = SpecifierSet(f"=={old_version}")
    ranged = SpecifierSet(f">={old_version},<{_upper_bound(old_version)}")

    def rewrite_requirement(match: re.Match[str]) -> str:
        original = match.group(0)
        parsed = Requirement(original)
        if parsed.url or parsed.specifier not in (exact, ranged):
            raise ValueError(f"unsupported or stale ExecuTorch requirement: {original}")
        constraints = match["constraints"]
        updated = re.sub(
            r"(?P<operator>===|==|>=|<=|~=|!=|<|>)(?P<space>\s*)(?P<version>[^\s,;]+)",
            lambda clause: (
                clause["operator"]
                + clause["space"]
                + (
                    _upper_bound(new_version)
                    if clause["operator"] == "<"
                    else new_version
                )
            ),
            constraints,
        )
        start = match.start("constraints") - match.start()
        return original[:start] + updated + original[start + len(constraints) :]

    pending = []
    for path in _pin_site_paths():
        try:
            text = path.read_text(encoding="utf-8")
            if path == _VERSIONS_FILE:
                updated = _rewrite_versions(text, new_version, new_commit)
            else:
                updated, count = _REQUIREMENT.subn(rewrite_requirement, text)
                if not count and old_commit not in text:
                    raise ValueError("no current version or source pin found")
                updated = updated.replace(old_commit, new_commit)
            pending.append((path, text, updated))
        except (
            OSError,
            UnicodeError,
            ValueError,
            InvalidRequirement,
            yaml.YAMLError,
        ) as error:
            raise SystemExit(f"cannot update pin site {path}: {error}") from error

    changed = False
    for path, original, updated in pending:
        if updated != original:
            try:
                path.write_text(updated, encoding="utf-8")
            except OSError as error:
                raise SystemExit(f"cannot write pin site {path}: {error}") from error
            changed = True
    return changed


def _index_args(track: str, channel: str) -> list[str]:
    if track == "nightly":
        return [
            "--pre",
            "--index-url",
            f"https://download.pytorch.org/whl/nightly/{channel}",
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=("nightly", "stable"), default="nightly")
    parser.add_argument(
        "--channel",
        default="cu130",
        help="nightly CUDA channel for version selection and provenance",
    )
    parser.add_argument(
        "--allow-downgrade",
        action="store_true",
        help="explicitly authorize proposing a lower version; builds and tests still must pass",
    )
    args = parser.parse_args(argv)
    if args.track == "nightly" and args.channel not in {"cu130", "cu132"}:
        parser.error("supported TensorRT nightly CUDA channels are cu130 and cu132")
    index_args = _index_args(args.track, args.channel)
    target = pick_target(available_versions(index_args), args.track)
    current = read_pin("__executorch_version__")
    if target == current:
        print(f"executorch pin is already at the newest {args.track} version {current}")
        return 0
    if not args.allow_downgrade and Version(target) < Version(current):
        print(
            f"target {target} sorts below current pin {current}; refusing downgrade. Pass --allow-downgrade for a deliberate re-pin.",
            file=sys.stderr,
        )
        return 1
    commit = wheel_git_version(target, index_args)
    if write_pins(target, commit):
        print(f"moved executorch pin {current} -> {target} (commit {commit})")
    else:
        print("nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
