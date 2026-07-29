"""CLI for the test-suite manifest:  python -m tests.ci {list,show,run,matrix,doctor}

list                       all suites, tiers, lanes, variants
show <name>                a suite's resolved command per variant
run <name> [opts] [-- ...]  run one suite (the call CI + just both make)
matrix [--lane|--tier]     JSON matrix `include` for GitHub Actions
doctor                     validate the manifest (CI lints this)
"""

from __future__ import annotations

import argparse
import json
import sys

from .runner import REPO_ROOT, describe, junit_path, matrix, run_suite, select
from .suites import SUITES, by_name


def _cmd_list(_: argparse.Namespace) -> int:
    width = max(len(s.name) for s in SUITES)
    print(
        f"{'SUITE'.ljust(width)}  TIER  LANES                  VARIANTS         PLATFORMS"
    )
    for s in SUITES:
        print(
            f"{s.name.ljust(width)}  {s.tier:<4}  "
            f"{','.join(s.lanes):<21}  {','.join(s.variants):<15}  {','.join(s.platforms)}"
        )
    print(
        f"\n{len(SUITES)} suites.  "
        f"Run one:  python -m tests.ci run <suite>  (or `just suite <suite>`)"
    )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    s = by_name(args.name)
    print(f"# {s.name}  (tier={s.tier}, lanes={','.join(s.lanes)})")
    for var in s.variants:
        print(f"\n## variant: {var}   junit: {junit_path(s).name}")
        print(describe(s, var))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    s = by_name(args.name)
    variants = [args.variant] if args.variant else list(s.variants)
    if args.variant and args.variant not in s.variants:
        print(
            f"::warning::{s.name} does not run on variant {args.variant!r}; "
            f"it runs on {s.variants}",
            file=sys.stderr,
        )
        return 0
    rc = 0
    for var in variants:
        rc = run_suite(s, var, dry_run=args.dry_run, extra=args.pytest_args) or rc
    return rc


def _cmd_run_lane(args: argparse.Namespace) -> int:
    """Run every suite in a lane/tier, continuing past failures (so one consolidated
    report sees them all). Returns non-zero if any suite failed."""
    jobs = select(
        lane=args.lane, tier=args.tier, variant=args.variant, platform=args.platform
    )
    if not jobs:
        print("::warning::no suites match the given filters", file=sys.stderr)
        return 0
    rc = 0
    for s, var in jobs:
        rc = run_suite(s, var, dry_run=args.dry_run) or rc
    return rc


def _cmd_matrix(args: argparse.Namespace) -> int:
    include = matrix(
        lane=args.lane, tier=args.tier, variant=args.variant, platform=args.platform
    )
    if not include:
        print("::warning::matrix is empty for the given filters", file=sys.stderr)
    print(json.dumps({"include": include}))
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    """Static checks CI can gate on: unique names, unique junit paths, valid setup
    steps, declared cwd dirs exist, every suite is reachable by some lane."""
    problems: list[str] = []
    names = [s.name for s in SUITES]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        problems.append(f"duplicate suite names: {sorted(dupes)}")

    junits = [junit_path(s).name for s in SUITES]
    jdupes = {j for j in junits if junits.count(j) > 1}
    if jdupes:
        problems.append(f"colliding junit paths: {sorted(jdupes)}")

    valid_setup = {"hub", "executorch", "cuda-core", "mpi"}
    for s in SUITES:
        for step in s.setup:
            if step not in valid_setup:
                problems.append(f"{s.name}: unknown setup step {step!r}")
        if not s.lanes:
            problems.append(f"{s.name}: belongs to no lane")
        if not s.variants:
            problems.append(f"{s.name}: runs on no variant")
        if not s.platforms:
            problems.append(f"{s.name}: runs on no platform")
        cwd = REPO_ROOT / s.cwd
        if not cwd.is_dir():
            problems.append(f"{s.name}: cwd {s.cwd} does not exist")
        for var in s.variants:
            if var not in (s.overrides.keys() | {"standard", "rtx"}):
                problems.append(f"{s.name}: bad variant {var!r}")

    # Every suite should be exercised by some lane and some tier path.
    if problems:
        for p in problems:
            print(f"✗ {p}", file=sys.stderr)
        print(f"\n{len(problems)} manifest problem(s).", file=sys.stderr)
        return 1
    print(
        f"✓ manifest OK — {len(SUITES)} suites, "
        f"{len(set(junits))} unique junit paths, no collisions."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tests.ci", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list all suites").set_defaults(fn=_cmd_list)

    sp = sub.add_parser("show", help="show a suite's resolved command")
    sp.add_argument("name")
    sp.set_defaults(fn=_cmd_show)

    sp = sub.add_parser("run", help="run one suite")
    sp.add_argument("name")
    sp.add_argument("--variant", choices=("standard", "rtx"))
    sp.add_argument(
        "--dry-run", action="store_true", help="print the command, don't run"
    )
    sp.add_argument(
        "pytest_args",
        nargs="*",
        help="extra args forwarded to pytest " "(use `-- -k foo`)",
    )
    sp.set_defaults(fn=_cmd_run)

    sp = sub.add_parser(
        "run-lane", help="run every suite in a lane/tier, past failures"
    )
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--lane", choices=("fast", "full", "nightly", "python-only"))
    g.add_argument("--tier", choices=("l0", "l1", "l2"))
    sp.add_argument("--variant", choices=("standard", "rtx"))
    sp.add_argument("--platform", choices=("linux-x86_64", "windows"))
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(fn=_cmd_run_lane)

    sp = sub.add_parser("matrix", help="emit a GitHub Actions matrix as JSON")
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--lane", choices=("fast", "full", "nightly", "python-only"))
    g.add_argument("--tier", choices=("l0", "l1", "l2"))
    sp.add_argument("--variant", choices=("standard", "rtx"))
    sp.add_argument("--platform", choices=("linux-x86_64", "windows"))
    sp.set_defaults(fn=_cmd_matrix)

    sub.add_parser("doctor", help="validate the manifest").set_defaults(fn=_cmd_doctor)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
