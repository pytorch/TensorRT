#!/usr/bin/env python3
"""Move the ExecuTorch pin to the newest wheel published on an index.

The pin is two coupled facts, spread across the tree but sourced from
``dev_dep_versions.yml``: ``__executorch_version__`` selects the wheel the delegate is
built to sit beside, and ``__executorch_commit__`` selects the tree it compiles from.
They must name one ExecuTorch, so this script never guesses the commit: it reads it from
the chosen wheel's own ``executorch/version.py``, which is the same provenance
``tests/py/dynamo/executorch/test_executorch_pin.py`` checks the pins against.

The daily workflow runs this, then opens a pull request when the pin moved. The existing
pin guards and the executorch end-to-end lane run on that pull request, so "the newest
wheel that actually works" is decided by the same gate a human bump goes through, not
re-implemented here. A nightly that did not publish leaves the newest version unchanged,
so the run rewrites nothing and opens nothing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.version import InvalidVersion, Version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSIONS_FILE = _REPO_ROOT / "dev_dep_versions.yml"

# A wheel version on the PyTorch index carries a local label naming its CUDA build, for
# example ``1.5.0.devYYYYMMDD+cu130``. The pin omits it so one pin serves every CUDA row.
_LOCAL_LABEL = re.compile(r"\+.*$")


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def read_pin(field: str) -> str:
    """Return a pinned value from ``dev_dep_versions.yml``."""
    text = _VERSIONS_FILE.read_text(encoding="utf-8")
    match = re.search(rf'^{field}:\s*"?([^"\s]+)"?\s*$', text, re.MULTILINE)
    if match is None:
        raise SystemExit(f"{field} is not set in {_VERSIONS_FILE.name}")
    return match.group(1)


def available_versions(index_args: list[str]) -> list[str]:
    """Every ExecuTorch version the index offers.

    ``pip index versions`` prints one ``Available versions:`` line. Parsing that is stable
    and needs no network code here; the workflow passes the index the same way every other
    install in the tree does.
    """
    out = _run(
        [sys.executable, "-m", "pip", "index", "versions", "executorch", *index_args]
    )
    match = re.search(r"^\s*Available versions:\s*(.+)$", out, re.MULTILINE)
    if match is None:
        raise SystemExit("pip index versions printed no Available versions line")
    return [v.strip() for v in match.group(1).split(",") if v.strip()]


def pick_target(versions: list[str], track: str) -> str:
    """The newest version on the wanted track.

    ``nightly`` takes the newest dated dev build; ``stable`` takes the newest final
    release, ignoring dev builds and release candidates. Ordering is PEP 440, not string
    order, so a newer dev date on the same line sorts above an older one correctly.
    """
    parsed: list[tuple[Version, str]] = []
    for raw in versions:
        try:
            version = Version(raw)
        except InvalidVersion:
            continue
        # A nightly is a dated dev build. is_prerelease is also true for release
        # candidates, and an rc sorts above every dev of the same line under PEP 440, so
        # filtering on it would let the first rc on the index silently become the nightly
        # pin. Match the dev segment itself instead.
        if track == "nightly" and version.dev is None:
            continue
        if track == "stable" and (version.is_prerelease or version.is_devrelease):
            continue
        parsed.append((version, raw))
    if not parsed:
        raise SystemExit(f"no executorch version on the index matches track {track!r}")
    newest = max(parsed, key=lambda pair: pair[0])[1]
    return _LOCAL_LABEL.sub("", newest)


def wheel_git_version(version: str, index_args: list[str]) -> str:
    """The source commit the chosen wheel records for itself.

    Every published wheel writes ``git_version`` into ``executorch/version.py``. Reading it
    from the wheel is what keeps the two pins naming one ExecuTorch. A wheel built without
    git provenance records ``None`` and must not become a pin, so that is an error, not a
    guess.
    """
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
        if not wheels:
            raise SystemExit(
                f"pip download produced no wheel for executorch=={version}"
            )
        with zipfile.ZipFile(wheels[0]) as archive:
            source = archive.read("executorch/version.py").decode("utf-8")
    match = re.search(r"""git_version[^=]*=\s*['"]([0-9a-f]{40})['"]""", source)
    if match is None:
        raise SystemExit(
            f"executorch=={version} records no source commit, so the pin would name a "
            "wheel whose provenance cannot be checked"
        )
    return match.group(1)


def _upper_bound(version: str) -> str:
    """The exclusive upper bound a range site pairs with the pin, next minor of its line.

    ``tests/py/dynamo/executorch/test_executorch_pin.py`` derives the same bound from the
    same two fields, so a range this writes and the range the guard expects agree by the
    same rule rather than by coincidence.
    """
    major, minor = version.split(".")[:2]
    return f"{major}.{int(minor) + 1}"


# The only files that carry the pin as a real pin. A tree-wide literal replace was safe while the
# pin was a dated dev string, because "1.5.0.dev20260825" appears nowhere else, but it corrupts the
# tree the moment the pin is a plain release like "1.4.1": that token also lives in unrelated
# requirements (for example pandocfilters>=1.4.1 in committed notebooks) and, worst, in uv.lock,
# whose entries are content addressed, so rewriting the version inside a wheel URL while its hash and
# size stay behind is a guaranteed install failure. So restrict the rewrite to the sites the guard in
# tests/py/dynamo/executorch/test_executorch_pin.py enumerates, which are the only sites that are
# actually pins. A new legitimate site must be added here and to that guard together.
_PIN_SITES = (
    ".github/workflows/executorch-build-linux.yml",
    ".github/workflows/executorch-test-linux.yml",
    "MODULE.bazel",
    "docker/MODULE.bazel.docker",
    "docker/MODULE.bazel.ngc",
    "justfile",
    "py/torch-tensorrt-executorch-runtime/README.md",
    "py/torch-tensorrt-executorch-runtime/pyproject.toml",
    "toolchains/ci_workspaces/MODULE.bazel.tmpl",
    "examples/executorch_reference_runner/README.md",
)


def _pin_site_paths() -> list[Path]:
    tracked = set(_run(["git", "-C", str(_REPO_ROOT), "ls-files"]).splitlines())
    # An entry that is no longer tracked is a stale list, not an absent pin, so refuse rather than
    # skip it. Skipping rewrites the other sites and returns success, and the workflow's gate is
    # `git diff --quiet`, which detects change and not coherence, so the bot would open a pull
    # request whose unrewritten site still names the old pin. The likely cause is a pin site being
    # renamed without this list following it.
    missing = sorted(name for name in _PIN_SITES if name not in tracked)
    if missing:
        raise SystemExit(
            "these pin sites are not tracked by git, so the pin cannot be rewritten "
            f"consistently: {missing}. Update _PIN_SITES, and the matching list in "
            "tests/py/dynamo/executorch/test_executorch_pin.py, if a file moved."
        )
    paths = [_REPO_ROOT / name for name in _PIN_SITES]
    # dev_dep_versions.yml is the source of truth and is rewritten too; it is not in _PIN_SITES
    # because the guard reads it rather than counting it as a downstream pin site.
    paths.append(_VERSIONS_FILE)
    return paths


def write_pins(new_version: str, new_commit: str) -> bool:
    """Rewrite the pin to the new version and commit at the known pin sites.

    The rewrite is restricted to the sites the guard enumerates (see _PIN_SITES), and within them it
    matches only a requirement on the executorch distribution. It is NOT a tree-wide literal
    replace, and it is not a bare version match either: a plain release like "1.4.1" appears in
    unrelated requirements, in content-addressed uv.lock entries, and in other pins inside the pin
    files themselves. Neither setup.py nor uv.lock is a pin site; they are left for their own tooling
    to regenerate. Returns whether anything changed.
    """
    old_version = read_pin("__executorch_version__")
    old_commit = read_pin("__executorch_commit__")
    if (new_version, new_commit) == (old_version, old_commit):
        return False

    # A single regex pass, so a freshly written new version cannot be matched again. A plain
    # text.replace of the bare version doubles the tail when the old version is a prefix of the new
    # one (1.5.0 -> 1.5.0.post1 would yield 1.5.0.post1.post1), because the second replace re-hits
    # what the first just wrote. The trailing boundary avoids that, and the range is handled by the
    # same alternation so its own version is not rewritten twice.
    #
    # Anchored on what names the pin, because a bare version is not distinctive enough to identify
    # one. Downstream sites all spell it as a requirement on the executorch distribution, while the
    # same files carry unrelated versions that are lexically identical: each MODULE.bazel has
    # bazel_dep(name = "bazel_skylib", version = "1.7.1"), so a pin of 1.7.1 rewrote that too.
    # Neither a leading nor a trailing character-class boundary helps there, since the character
    # before the version is a quote, and restricting the rewrite to known pin files does not help
    # either, because those are the very files holding the bystanders. dev_dep_versions.yml is the
    # exception: it is the source of truth and states the version as a YAML key rather than a
    # requirement, so it gets its own alternative.
    #
    # Every operator the guard in tests/py/dynamo/executorch/test_executorch_pin.py treats as a pin,
    # with its optional spaces, not just the two spellings the tree happens to use today. A site the
    # guard counts but this rewriter skips is the worst shape available: the bump leaves it on the old
    # version, and the guard then fails the generated pull request as a pin mismatch rather than as an
    # operator the rewriter cannot see.
    #
    # The name needs a left boundary of its own, or "my-executorch==<pin>" matches on its tail.
    # Distribution names normalise hyphens and underscores together, so exclude both, plus a dot.
    old_upper = _upper_bound(old_version)
    new_upper = _upper_bound(new_version)
    old_range_tail = f"{old_version},<{old_upper}"
    version_token = re.compile(
        r"(?P<lead>(?<![0-9A-Za-z._-])executorch ?(?:===|==|>=|<=|~=|!=|<|>) ?"
        r"|__executorch_version__:\s*\"?)(?:"
        + re.escape(old_range_tail)
        + r"|"
        + re.escape(old_version)
        + r")(?![0-9A-Za-z.+_-])"
    )

    def _sub_version(match: re.Match[str]) -> str:
        lead = match.group("lead")
        if match.group(0).endswith(f",<{old_upper}"):
            return f"{lead}{new_version},<{new_upper}"
        return f"{lead}{new_version}"

    changed = False
    for path in _pin_site_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        if old_version not in text and old_commit not in text:
            continue
        updated = version_token.sub(_sub_version, text)
        updated = updated.replace(old_commit, new_commit)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed = True

    if read_pin("__executorch_version__") != new_version:
        raise SystemExit(
            "the version pin did not take; dev_dep_versions.yml is unchanged"
        )
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
        help="nightly CUDA channel to read versions and provenance from",
    )
    parser.add_argument(
        "--allow-downgrade",
        action="store_true",
        help=(
            "move the pin even when the target sorts below the current pin. Off by default so a "
            "regressed index, or switching --track from nightly to stable, cannot silently walk "
            "the pin backwards onto a version this delegate cannot build against."
        ),
    )
    args = parser.parse_args(argv)

    index_args = _index_args(args.track, args.channel)
    target = pick_target(available_versions(index_args), args.track)
    current = read_pin("__executorch_version__")
    if target == current:
        print(f"executorch pin is already at the newest {args.track} version {current}")
        return 0
    # Never move backwards unless asked. pick_target returns the newest on the track, but "newest"
    # regresses when the index drops the current line or when --track flips from nightly to stable,
    # whose newest final release can sort below a dated nightly. The pin then lands on a version with
    # no standalone libexecutorch.so and no CUDA build, which the delegate cannot link, and the run
    # would still report success. Refuse it, and require an explicit opt-in for a deliberate re-pin.
    if not args.allow_downgrade and Version(target) < Version(current):
        print(
            f"target {target} sorts below the current pin {current}; refusing to move the pin "
            "backwards. Pass --allow-downgrade to override for a deliberate re-pin."
        )
        return 0

    commit = wheel_git_version(target, index_args)
    if write_pins(target, commit):
        print(f"moved executorch pin {current} -> {target} (commit {commit})")
    else:
        print("nothing to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
