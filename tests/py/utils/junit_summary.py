#!/usr/bin/env python3
"""Aggregate JUnit XML results from a directory into one consolidated report.

The local test tiers (``just tests-report ...``) and CI both write one JUnit
XML per pytest suite. When several suites run back-to-back, failures scroll
off-screen and a later passing suite's exit code can mask an earlier failure —
so failures get missed. This reads every XML in the results dir and prints a
single consolidated report (independent of any exit code). Exits non-zero if
anything failed or errored.

Two output modes:
  (default)  a color-coded, human-friendly terminal report.
  --agent    a plain Markdown report built for handing to an AI agent: every
             failure with its exact pytest node id, test file, the JUnit XML
             path (to read the full traceback), the message + (capped) detail,
             and a copy-paste repro command.

Usage:
    python3 tests/py/utils/junit_summary.py [RESULTS_DIR] [--agent]

RESULTS_DIR defaults to $RUNNER_TEST_RESULTS_DIR, else $TMPDIR/trt_test_results.
Color (default mode) is on for a TTY; honors NO_COLOR / FORCE_COLOR. Stdlib-only.
"""

import argparse
import glob
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_AGENT_DETAIL_MAX_LINES = 40  # cap per-failure traceback in --agent mode

# ── color ─────────────────────────────────────────────────────────────────────


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


_USE_COLOR = _color_enabled()
_CODES = {
    "reset": "0",
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
    "grey": "90",
}


def c(text: str, *styles: str) -> str:
    if not _USE_COLOR or not styles:
        return text
    codes = ";".join(_CODES[s] for s in styles)
    return f"\033[{codes}m{text}\033[0m"


def _visible_len(s: str) -> int:
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            j = s.find("m", i)
            if j != -1:
                i = j + 1
                continue
        out += 1
        i += 1
    return out


def _pad(s: str, width: int) -> str:
    gap = width - _visible_len(s)
    return s + " " * gap if gap > 0 else s


# ── parsing ─────────────────────────────────────────────────────────────────


@dataclass
class Suite:
    label: str
    xml_path: str
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors + self.skipped

    @property
    def bad(self) -> int:
        return self.failed + self.errors


@dataclass
class Failure:
    kind: str  # FAIL | ERROR | PARSE
    test: str  # classname::name (exact pytest node identity)
    message: str  # one-line headline
    detail: str  # full traceback text
    file: str  # test file path (may be empty)
    suite: str  # suite label
    xml_path: str  # full path to the JUnit XML (read for full detail)


@dataclass
class Report:
    results_dir: str
    suites: list = field(default_factory=list)
    failures: list = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(s.passed for s in self.suites)

    @property
    def failed(self) -> int:
        return sum(s.failed for s in self.suites)

    @property
    def errors(self) -> int:
        return sum(s.errors for s in self.suites)

    @property
    def skipped(self) -> int:
        return sum(s.skipped for s in self.suites)

    @property
    def total(self) -> int:
        return sum(s.total for s in self.suites)

    @property
    def bad(self) -> int:
        return self.failed + self.errors


def _suite_label(xml_path: str) -> str:
    name = os.path.basename(xml_path)
    for suffix in ("_tests_results.xml", "_test_results.xml", "_results.xml", ".xml"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _first_line(text: str, fallback: str = "") -> str:
    text = (text or "").strip()
    return text.splitlines()[0].strip() if text else fallback


def _repro(failure: "Failure") -> str:
    """A copy-paste pytest command that re-runs (at least) the failing test."""
    name = failure.test.split("::")[-1]
    knode = name.split("[")[0] or name  # drop parametrization for -k
    target = failure.file or "tests/py"
    return f"uv run --no-sync pytest {target} -k {knode!r} -n0"


def parse(results_dir: str) -> Report:
    report = Report(results_dir=results_dir)
    for xml in sorted(glob.glob(os.path.join(results_dir, "*.xml"))):
        suite = Suite(label=_suite_label(xml), xml_path=xml)
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError as e:
            suite.errors += 1
            report.failures.append(
                Failure(
                    "PARSE",
                    os.path.basename(xml),
                    f"could not parse XML: {e}",
                    "",
                    "",
                    suite.label,
                    xml,
                )
            )
            report.suites.append(suite)
            continue

        for ts in root.iter("testsuite"):
            for tc in ts.iter("testcase"):
                classname = tc.get("classname", "")
                name = tc.get("name", "")
                test = f"{classname}::{name}" if classname else name
                file_attr = tc.get("file", "")
                fa, er, sk = tc.find("failure"), tc.find("error"), tc.find("skipped")
                if fa is not None:
                    suite.failed += 1
                    report.failures.append(
                        Failure(
                            "FAIL",
                            test,
                            _first_line(fa.get("message") or fa.text, "failed"),
                            (fa.text or "").strip(),
                            file_attr,
                            suite.label,
                            xml,
                        )
                    )
                elif er is not None:
                    suite.errors += 1
                    report.failures.append(
                        Failure(
                            "ERROR",
                            test,
                            _first_line(er.get("message") or er.text, "errored"),
                            (er.text or "").strip(),
                            file_attr,
                            suite.label,
                            xml,
                        )
                    )
                elif sk is not None:
                    suite.skipped += 1
                else:
                    suite.passed += 1
        report.suites.append(suite)
    return report


# ── pretty (human) rendering ──────────────────────────────────────────────────


def _truncate(s: str, width: int) -> str:
    if width <= 1 or _visible_len(s) <= width:
        return s
    return s[: width - 1] + "…"


def _count(n: int, label: str, style: str) -> str:
    if n == 0:
        return c(f"{n:>4} {label}", "grey")
    return c(f"{n:>4}", style, "bold") + " " + c(label, style)


def render_pretty(report: Report) -> int:
    width = min(shutil.get_terminal_size((100, 24)).columns, 100)
    rule = c("─" * width, "grey")

    print()
    print(c(" Torch-TensorRT · Test Report ", "bold", "cyan"))
    print(c(f" {report.results_dir}", "grey"))

    if not report.suites:
        print()
        print(c("  no JUnit XML files found — did the run write results?", "yellow"))
        print()
        return 1

    print()
    print(rule)
    label_w = min(
        max(max((_visible_len(s.label) for s in report.suites), default=20), 12),
        width - 40,
    )
    for s in sorted(report.suites, key=lambda s: (s.bad == 0, s.label)):
        if s.bad:
            icon, lstyle = c("✗", "red", "bold"), "red"
        elif s.passed == 0 and s.skipped:
            icon, lstyle = c("⊘", "yellow"), "yellow"
        else:
            icon, lstyle = c("✓", "green"), "green"
        cells = "  ".join(
            [
                _count(s.passed, "pass", "green"),
                _count(s.failed, "fail", "red"),
                _count(s.errors, "err", "magenta"),
                _count(s.skipped, "skip", "yellow"),
            ]
        )
        print(f"  {icon}  {_pad(c(s.label, lstyle), label_w)}  {cells}")
    print(rule)

    totals = "  ".join(
        [
            _count(report.passed, "pass", "green"),
            _count(report.failed, "fail", "red"),
            _count(report.errors, "err", "magenta"),
            _count(report.skipped, "skip", "yellow"),
        ]
    )
    meta = c(f"({report.total} tests · {len(report.suites)} suites)", "grey")
    print(f"  {_pad(c('TOTAL', 'bold'), label_w + 3)}{totals}  {meta}")

    if report.failures:
        print()
        print(c(f" Failures ({len(report.failures)})", "bold", "red"))
        print()
        num_w = len(str(len(report.failures)))
        for i, f in enumerate(report.failures, 1):
            tag = c(f"{f.kind:<5}", "magenta" if f.kind == "ERROR" else "red", "bold")
            print(f"  {c(f'{i:>{num_w}}.', 'grey')} {tag} {c(f.suite, 'grey')}")
            print(f"  {' ' * num_w}  {c('•', 'grey')} {c(f.test, 'bold')}")
            if f.message:
                print(
                    f"  {' ' * num_w}    {c(_truncate(f.message, width - num_w - 6), 'yellow')}"
                )
            print(f"  {' ' * num_w}    {c('↻ ' + _repro(f), 'grey')}")
            print()

    # Footer: point at the raw results + the agent report.
    print(c(" JUnit XMLs:", "grey"), c(report.results_dir, "grey"))
    print(
        c(" For a paste-to-Claude report:", "grey"),
        c("just test-summary --agent", "cyan"),
    )
    if report.bad:
        print(
            c(f" ✗ {report.failed} failed · {report.errors} errored ", "bold", "red")
            + c(f"  ({report.total} tests · {len(report.suites)} suites)", "grey")
        )
    else:
        print(
            c(f" ✓ all {report.passed} passed ", "bold", "green")
            + c(f"  ({report.total} tests · {len(report.suites)} suites)", "grey")
        )
    print()
    return 1 if report.bad else 0


# ── agent (Markdown) rendering ────────────────────────────────────────────────


def render_agent(report: Report) -> int:
    out = print  # plain, no color, capturable

    out("# Torch-TensorRT test report")
    out("")
    out(f"- results dir: `{report.results_dir}`")
    out(
        f"- totals: {report.passed} passed, {report.failed} failed, "
        f"{report.errors} errored, {report.skipped} skipped "
        f"({report.total} tests across {len(report.suites)} suites)"
    )
    if not report.suites:
        out("")
        out("No JUnit XML files found — the run did not write results.")
        return 1
    out("- to read a full traceback, open the `junit` path listed for the failure")
    out(
        "- repro commands use `-n0` (serial); drop `-k` to run the whole file, "
        "lower `-n`/free GPU memory if a test OOMs"
    )
    out("")

    # Suite table (compact, Markdown).
    out("## Suites")
    out("")
    out("| suite | passed | failed | errors | skipped |")
    out("|---|--:|--:|--:|--:|")
    for s in sorted(report.suites, key=lambda s: (s.bad == 0, s.label)):
        mark = "" if s.bad == 0 else " ⚠"
        out(
            f"| `{s.label}`{mark} | {s.passed} | {s.failed} | {s.errors} | {s.skipped} |"
        )
    out("")

    if not report.failures:
        out("## Result: all green ✅")
        return 0

    out(f"## Failures ({len(report.failures)})")
    out("")
    for i, f in enumerate(report.failures, 1):
        out(f"### {i}. [{f.kind}] `{f.test}`")
        out(f"- suite: `{f.suite}`")
        if f.file:
            out(f"- file: `{f.file}`")
        out(f"- junit: `{f.xml_path}`")
        out(f"- repro: `{_repro(f)}`")
        if f.message:
            out(f"- message: {f.message}")
        if f.detail:
            lines = f.detail.splitlines()
            clipped = lines[:_AGENT_DETAIL_MAX_LINES]
            out("- detail:")
            out("```")
            for ln in clipped:
                out(ln)
            if len(lines) > _AGENT_DETAIL_MAX_LINES:
                out(
                    f"... [{len(lines) - _AGENT_DETAIL_MAX_LINES} more lines; "
                    f"read the full traceback in the junit XML above]"
                )
            out("```")
        out("")
    return 1


def _results_dir(value) -> str:
    if value:
        return value
    return os.environ.get("RUNNER_TEST_RESULTS_DIR") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "trt_test_results"
    )


def main(argv: list) -> int:
    p = argparse.ArgumentParser(description="Summarize JUnit XML test results.")
    p.add_argument("results_dir", nargs="?", help="dir of JUnit *.xml files")
    p.add_argument(
        "--agent",
        "--claude",
        action="store_true",
        dest="agent",
        help="emit a plain Markdown report for handing to an AI agent",
    )
    args = p.parse_args(argv[1:])
    report = parse(_results_dir(args.results_dir))
    return render_agent(report) if args.agent else render_pretty(report)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
