"""The pin updater has to be trusted to run unattended and open a pull request, so its
parts are tested the way they fail in practice: version ordering that is not string order,
a wheel that forgot its provenance, and a rewrite that has to leave the tree in exactly the
state the pin guard demands. Every test runs the real function; none restates it.

Text and metadata only, like test_executorch_pin.py: no network, no GPU, no ExecuTorch. The
two functions that reach the index (available_versions, wheel_git_version) are exercised
against captured output and a synthesized wheel, so this file runs on the lint runner.

Every version here is a fake far-past date (year 2020) and every commit is an obvious
marker, never a real pin. The updater bumps the pin by replacing the old literal everywhere
it appears in the tree, so a real pin value living in this file would be rewritten by a
bump, quietly changing the fixtures. Synthetic values can never equal the live pin, so a
bump never touches this file. test_write_pins_updates_new_sites_but_never_its_own_source
holds that line.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / ".github" / "scripts" / "update_executorch_pin.py"
_SELF = "tests/py/dynamo/executorch/test_update_executorch_pin.py"


def _load():
    spec = importlib.util.spec_from_file_location("update_executorch_pin", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


updater = _load()


# The shape of a `pip index versions executorch` run: dated nightlies of one line, out of
# order so a test that passes under string sorting still fails here. The dates are fake and
# far in the past so they can never equal the live pin.
_NIGHTLY_LIST = [
    "1.5.0.dev20200101+cu130",
    "1.5.0.dev20200103+cu130",
    "1.5.0.dev20200102+cu130",
]
# A stable channel carries finals and the occasional release candidate; the updater must
# take the newest final, not the newer-looking prerelease. The line is a fake 0.9 that
# ExecuTorch will never ship, so it cannot equal a real stable pin either.
_STABLE_LIST = ["0.9.1", "0.9.2", "0.9.3", "0.9.4rc1"]


def test_pick_target_nightly_takes_the_newest_date_not_the_longest_string() -> None:
    # Distinct on purpose: if a tree-wide bump ever collapsed two of these into one, the
    # list would stop testing ordering, so fail loudly the moment they are not unique.
    assert len(set(_NIGHTLY_LIST)) == len(_NIGHTLY_LIST)
    assert updater.pick_target(_NIGHTLY_LIST, "nightly") == "1.5.0.dev20200103"


def test_pick_target_strips_the_cuda_local_label() -> None:
    # The pin serves every CUDA row, so the +cuXXX label the index carries must not survive
    # into the pin. A label left on would fail the guard's exact-match on every site.
    assert "+" not in updater.pick_target(_NIGHTLY_LIST, "nightly")


def test_pick_target_stable_ignores_dev_and_release_candidates() -> None:
    assert updater.pick_target(_STABLE_LIST, "stable") == "0.9.3"


def test_pick_target_nightly_ignores_finals() -> None:
    # A stable final on the nightly index is not a nightly; picking it would move the pin
    # off the dev line the delegate is built against.
    assert updater.pick_target(["0.9.3", "1.5.0.dev20200103"], "nightly") == (
        "1.5.0.dev20200103"
    )


def test_pick_target_nightly_ignores_release_candidates() -> None:
    # An rc is a prerelease that sorts above every dev of the same line under PEP 440, so a
    # filter that only excluded finals would let the first rc on the nightly index become the
    # pin. A nightly is a dated dev build, and an rc is not one.
    assert (
        updater.pick_target(["1.5.0.dev20200103+cu130", "1.5.0rc1+cu130"], "nightly")
        == "1.5.0.dev20200103"
    )


def test_pick_target_raises_when_nothing_matches_the_track() -> None:
    with pytest.raises(SystemExit):
        updater.pick_target(["0.9.3", "0.9.2"], "nightly")


def test_available_versions_parses_the_pip_line(monkeypatch) -> None:
    captured = (
        "executorch (1.5.0.dev20200103+cu130)\n"
        "Available versions: 1.5.0.dev20200103+cu130, 1.5.0.dev20200102+cu130\n"
        "  INSTALLED: 1.1.0\n"
        "  LATEST:    1.5.0.dev20200103+cu130\n"
    )
    monkeypatch.setattr(updater, "_run", lambda cmd: captured)
    assert updater.available_versions([]) == [
        "1.5.0.dev20200103+cu130",
        "1.5.0.dev20200102+cu130",
    ]


def _synthesize_wheel(path: Path, body: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("executorch/version.py", body)


def test_wheel_git_version_reads_the_recorded_commit(monkeypatch, tmp_path) -> None:
    commit = "deadbeef" * 5
    wheel = tmp_path / "executorch-1.5.0.dev20200103-py3-none-any.whl"
    _synthesize_wheel(
        wheel, f'__version__ = "1.5.0.dev20200103"\ngit_version = "{commit}"\n'
    )

    def fake_download(cmd):
        # The download call writes the wheel into the temp dir named right after --dest.
        dest = Path(cmd[cmd.index("--dest") + 1])
        (dest / wheel.name).write_bytes(wheel.read_bytes())
        return ""

    monkeypatch.setattr(updater, "_run", fake_download)
    assert updater.wheel_git_version("1.5.0.dev20200103", []) == commit


def test_wheel_git_version_rejects_a_wheel_without_provenance(
    monkeypatch, tmp_path
) -> None:
    # ExecuTorch writes git_version = None when built outside a git checkout. Such a wheel
    # carries nothing the pins can be checked against, so it must not become a pin.
    wheel = tmp_path / "executorch-1.5.0.dev20200103-py3-none-any.whl"
    _synthesize_wheel(wheel, "__version__ = '1.5.0.dev20200103'\ngit_version = None\n")

    def fake_download(cmd):
        dest = Path(cmd[cmd.index("--dest") + 1])
        (dest / wheel.name).write_bytes(wheel.read_bytes())
        return ""

    monkeypatch.setattr(updater, "_run", fake_download)
    with pytest.raises(SystemExit):
        updater.wheel_git_version("1.5.0.dev20200103", [])


def test_upper_bound_is_the_next_minor_of_the_line() -> None:
    # A nightly and a final both belong to the release line their first two fields name, so
    # the bound is the next minor either way. This mirrors the guard's own derivation, and
    # the 0.9 case checks the minor rolls to a two-digit number rather than to "0.:".
    assert updater._upper_bound("1.5.0.dev20200103") == "1.6"
    assert updater._upper_bound("0.9.1") == "0.10"
    assert updater._upper_bound("7.3.0") == "7.4"


def _worktree(tmp_path) -> Path:
    """A throwaway checkout of the repo so write_pins edits a real tree, not a copy that
    diverges from what git tracks. write_pins walks `git ls-files`, so the tree has to be a
    real checkout."""
    work = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(_REPO_ROOT), str(work)],
        check=True,
    )
    return work


# A synthetic bump target for the write_pins tests: a fake far-past nightly and an obvious
# marker commit, distinct from any live pin so the bump is a real change but never collides
# with a real value in the tree.
_FAKE_VERSION = "1.5.0.dev20200103"
_FAKE_COMMIT = "deadbeef" * 5


def test_write_pins_leaves_a_tree_the_guard_accepts(tmp_path, monkeypatch) -> None:
    # The one test that matters most: after a bump, the whole pin guard has to pass, because
    # that guard is what the generated pull request will be judged by. A rewrite that the
    # guard rejects would open a red pull request every night.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    assert updater.write_pins(_FAKE_VERSION, _FAKE_COMMIT) is True
    assert updater.read_pin("__executorch_version__") == _FAKE_VERSION
    assert updater.read_pin("__executorch_commit__") == _FAKE_COMMIT

    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/py/dynamo/executorch/test_executorch_pin.py",
            "-q",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "-o",
            "addopts=",
        ],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_write_pins_is_idempotent_when_already_current(tmp_path, monkeypatch) -> None:
    # A day with no new nightly must rewrite nothing, so the run opens no pull request. The
    # updater is safe to run every day and by hand.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    current_version = updater.read_pin("__executorch_version__")
    current_commit = updater.read_pin("__executorch_commit__")
    assert updater.write_pins(current_version, current_commit) is False

    diff = subprocess.run(
        ["git", "-C", str(work), "diff", "--quiet"], capture_output=True
    )
    assert diff.returncode == 0, "write_pins changed the tree when the pin was current"


def test_write_pins_updates_new_sites_but_never_its_own_source(
    tmp_path, monkeypatch
) -> None:
    # The updater rewrites only the enumerated pin sites, not every file that happens to contain
    # the version token. That is what stops a plain release pin from corrupting unrelated
    # requirements and content-addressed lock entries. Three properties are pinned here: an
    # enumerated site is updated, a brand new unenumerated site is left alone, and the updater's
    # own source comes out byte for byte identical.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    old_version = updater.read_pin("__executorch_version__")
    # An enumerated pin site: MODULE.bazel carries the version and must be rewritten.
    enumerated = work / "MODULE.bazel"
    assert old_version in enumerated.read_text()
    # An unenumerated file carrying the same token must NOT be rewritten, the way an unrelated
    # requirement or a uv.lock entry must be left behind.
    bystander = work / "some_new_requirements.txt"
    bystander.write_text(f"executorch=={old_version}\n")
    subprocess.run(["git", "-C", str(work), "add", str(bystander)], check=True)

    script = work / ".github" / "scripts" / "update_executorch_pin.py"
    test = work / _SELF
    script_before = script.read_bytes()
    test_before = test.read_bytes()

    assert updater.write_pins(_FAKE_VERSION, _FAKE_COMMIT) is True

    assert _FAKE_VERSION in enumerated.read_text()
    assert bystander.read_text() == f"executorch=={old_version}\n"
    assert script.read_bytes() == script_before
    assert test.read_bytes() == test_before


def test_write_pins_leaves_another_packages_matching_version_alone(
    tmp_path, monkeypatch
) -> None:
    # A version on its own does not identify a pin. Every MODULE.bazel here declares
    # bazel_dep(name = "bazel_skylib", version = "1.7.1"), so a bare-version rewrite moved that
    # too whenever the ExecuTorch pin happened to be 1.7.1, silently changing an unrelated
    # dependency. Enumerating the pin files does not prevent it, because those are the very files
    # holding the bystander, so the rewrite has to key on what names the pin.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    module_bazel = work / "MODULE.bazel"
    skylib = 'bazel_dep(name = "bazel_skylib", version = "1.7.1")'
    assert (
        skylib in module_bazel.read_text()
    ), "fixture stale: bazel_skylib is not pinned to 1.7.1"
    # Collide the ExecuTorch pin with bazel_skylib's version, which is the case a bare-version
    # rewrite cannot tell apart. The downstream sites move with it, so this is a coherent tree
    # pinned to that version rather than one where only the source of truth changed.
    #
    # The colliding version is spelled at runtime rather than written out, because the pin guard
    # greps the tree for "executorch==<digits>" to inventory the pin sites, and a literal here would
    # register this test file as a new site.
    collided = "1.7.1"
    bumped = "1.8.0"
    requirement = f"executorch=={collided}"
    old_version = updater.read_pin("__executorch_version__")
    versions = work / "dev_dep_versions.yml"
    versions.write_text(
        re.sub(
            r'^__executorch_version__: ".*"$',
            f'__executorch_version__: "{collided}"',
            versions.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        ),
        encoding="utf-8",
    )
    for name in updater._PIN_SITES:
        site = work / name
        site.write_text(
            site.read_text(encoding="utf-8").replace(old_version, collided),
            encoding="utf-8",
        )
    assert requirement in module_bazel.read_text()

    assert updater.write_pins(bumped, _FAKE_COMMIT) is True

    assert updater.read_pin("__executorch_version__") == bumped
    assert skylib in module_bazel.read_text(), (
        "write_pins rewrote bazel_skylib's version because it shared the ExecuTorch pin's "
        "version string"
    )
    assert f"executorch=={bumped}" in module_bazel.read_text()


def test_write_pins_moves_every_requirement_shape_the_guard_counts(
    tmp_path, monkeypatch
) -> None:
    # The rewriter and the pin guard have to agree on what a pin looks like. The guard counts any
    # comparison operator, with optional spaces, so a site written "executorch >= X" is a pin to the
    # guard and invisible to a rewriter that only knows "==" and ">=". The bump would leave that site
    # behind and the guard would then fail the generated pull request, reported as a pin mismatch
    # rather than as an operator the rewriter cannot see. None of these shapes is in the tree today,
    # which is exactly why the agreement needs a test rather than an example.
    #
    # The negative half matters as much: a distribution whose name merely ends in executorch is a
    # different package that happens to share a version.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    old_version = updater.read_pin("__executorch_version__")
    probe = work / "MODULE.bazel"
    moved = [
        f"executorch{operator}{old_version}"
        for operator in ("==", " == ", ">=", "~=", "===")
    ]
    kept = [
        f"my-executorch=={old_version}",
        f"not_executorch=={old_version}",
        f"torch-tensorrt=={old_version}",
    ]
    # One requirement per line, and each line checked on its own. A substring search over the whole
    # file cannot work here: "my-executorch==<pin>" contains "executorch==<pin>", so a `not in` over
    # the joined text is satisfied by the line that is supposed to be left alone.
    probe.write_text(
        probe.read_text(encoding="utf-8")
        + "\n"
        + "\n".join(f"# {line}" for line in moved + kept)
        + "\n",
        encoding="utf-8",
    )

    assert updater.write_pins(_FAKE_VERSION, _FAKE_COMMIT) is True

    lines = {
        line.lstrip("# ").strip()
        for line in probe.read_text(encoding="utf-8").splitlines()
    }
    for line in moved:
        assert line not in lines, (
            f"write_pins left {line!r} on the old version, but the pin guard counts that shape as "
            "a pin, so the bump would produce a tree the guard rejects"
        )
    for line in kept:
        assert line in lines, (
            f"write_pins rewrote {line!r}, which is a different distribution that merely shares "
            "the pin's version"
        )


def test_write_pins_refuses_an_untracked_pin_site(tmp_path, monkeypatch) -> None:
    # A _PIN_SITES entry that git does not track is a stale list, not an absent pin. Skipping it
    # rewrites every other site and returns success, and the workflow's gate is `git diff --quiet`,
    # which sees change rather than coherence, so the bot would open a pull request whose
    # unrewritten site still names the old pin. The likely cause is a rename that this list did not
    # follow, so fail loudly and name the file.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    subprocess.run(
        ["git", "-C", str(work), "mv", "justfile", "justfile.renamed"], check=True
    )

    with pytest.raises(SystemExit) as error:
        updater.write_pins(_FAKE_VERSION, _FAKE_COMMIT)
    assert "justfile" in str(error.value)

    # And nothing was rewritten on the way to the refusal.
    assert updater.read_pin("__executorch_version__") != _FAKE_VERSION


def test_main_refuses_to_move_the_pin_backwards(monkeypatch) -> None:
    # pick_target returns the newest on the track, but "newest" regresses when the index drops the
    # current line or when --track flips to stable. Moving the pin backwards lands it on a version
    # the delegate cannot build against, so main() must refuse it and touch nothing.
    monkeypatch.setattr(updater, "read_pin", lambda field: "1.5.0.dev20260825")
    monkeypatch.setattr(updater, "available_versions", lambda index_args: ["1.4.1"])
    monkeypatch.setattr(updater, "pick_target", lambda versions, track: "1.4.1")

    def fail_wheel(*args, **kwargs):
        raise AssertionError("wheel_git_version must not run on a refused downgrade")

    def fail_write(*args, **kwargs):
        raise AssertionError("write_pins must not run on a refused downgrade")

    monkeypatch.setattr(updater, "wheel_git_version", fail_wheel)
    monkeypatch.setattr(updater, "write_pins", fail_write)

    assert updater.main(["--track", "nightly"]) == 0


def test_main_allows_a_backwards_move_with_the_opt_in(monkeypatch) -> None:
    # The deliberate re-pin escape hatch: --allow-downgrade lets a lower target through, so the
    # run reaches the wheel read and the write. The write's ARGUMENTS are asserted, not just that it
    # ran: main() is the only place the version it picked and the commit it read from that wheel are
    # paired, so a main() that dropped the commit and wrote something else would satisfy a
    # ran-or-not check while producing exactly the split pin this whole file exists to prevent.
    monkeypatch.setattr(updater, "read_pin", lambda field: "1.5.0.dev20260825")
    monkeypatch.setattr(updater, "available_versions", lambda index_args: ["1.4.1"])
    monkeypatch.setattr(updater, "pick_target", lambda versions, track: "1.4.1")
    wheel_commit = "c" * 40
    wheel_calls: list[str] = []
    write_calls: list[tuple[str, str]] = []

    def note_wheel(version, index_args):
        wheel_calls.append(version)
        return wheel_commit

    def note_write(version, commit):
        write_calls.append((version, commit))
        return True

    monkeypatch.setattr(updater, "wheel_git_version", note_wheel)
    monkeypatch.setattr(updater, "write_pins", note_write)

    assert updater.main(["--track", "nightly", "--allow-downgrade"]) == 0
    assert wheel_calls == ["1.4.1"]
    assert write_calls == [("1.4.1", wheel_commit)]


def test_write_pins_handles_a_prefix_version_bump(tmp_path, monkeypatch) -> None:
    # When the old version is a prefix of the new one (1.5.0 -> 1.5.0.post1), a plain two-pass
    # text.replace doubles the tail at a range site: the range pass writes >=1.5.0.post1,<1.6, then
    # the bare pass re-hits the 1.5.0 inside it and yields >=1.5.0.post1.post1,<1.6, which packaging
    # rejects. Bare-version sites do not double, so the regression has to be checked at a range site.
    # The single regex pass must produce the new version exactly.
    work = _worktree(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", work)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", work / "dev_dep_versions.yml")

    old = updater.read_pin("__executorch_version__")
    upper = updater._upper_bound("1.5.0")
    # Build the requirement token from parts so this test file itself does not read as a pin site to
    # the repo-wide requirement guard, which greps for the literal executorch>=<digit>.
    pkg = "executorch"
    old_range_line = f"{pkg}>=1.5.0,<{upper}"
    new_range_line = f"{pkg}>=1.5.0.post1,<{upper}"
    # Force both a bare-version site and a range site onto a plain 1.5.0, so old_range matches.
    versions = work / "dev_dep_versions.yml"
    versions.write_text(versions.read_text().replace(old, "1.5.0"))
    range_site = next(p for p in updater._pin_site_paths() if p.name == "MODULE.bazel")
    range_site.write_text(range_site.read_text() + f"\n{old_range_line}\n")
    # Stage rather than commit: write_pins walks `git ls-files`, which already lists staged
    # changes, and a commit would need a git identity the CI runner does not configure.
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)

    assert updater.write_pins("1.5.0.post1", "d" * 40) is True

    assert updater.read_pin("__executorch_version__") == "1.5.0.post1"
    text = range_site.read_text()
    assert (
        "1.5.0.post1.post1" not in text
    ), "range site double-substituted the prefix bump"
    assert new_range_line in text, "range site was not rewritten to the new pin"


def test_wheel_git_version_downloads_only_binaries(monkeypatch, tmp_path) -> None:
    # pip download runs an sdist's setup.py, and this runs in the pin-bump job that holds a
    # write-scoped token. The download must be wheel-only so a poisoned sdist on the index cannot
    # execute code there. The script only reads a wheel anyway.
    commit = "deadbeef" * 5
    wheel = tmp_path / "executorch-1.5.0.dev20200103-py3-none-any.whl"
    _synthesize_wheel(
        wheel, f'__version__ = "1.5.0.dev20200103"\ngit_version = "{commit}"\n'
    )
    seen = {}

    def fake_download(cmd):
        seen["cmd"] = cmd
        dest = Path(cmd[cmd.index("--dest") + 1])
        (dest / wheel.name).write_bytes(wheel.read_bytes())
        return ""

    monkeypatch.setattr(updater, "_run", fake_download)
    updater.wheel_git_version("1.5.0.dev20200103", [])
    assert "--only-binary=:all:" in seen["cmd"], seen["cmd"]
