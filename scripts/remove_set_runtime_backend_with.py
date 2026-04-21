#!/usr/bin/env python3
"""Remove obsolete ``with *.runtime.set_runtime_backend(...):`` blocks and dedent bodies."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WITH_RE = re.compile(
    r"^([ \t]*)with (torchtrt|torch_tensorrt|torch_trt)\.runtime\.set_runtime_backend\([^)]*\):\s*(#.*)?\s*$"
)


def visual_indent(line: str) -> tuple[int, int]:
    """Return (index after leading ws, visual column of next char) for spaces/tabs."""
    i = 0
    vis = 0
    n = len(line)
    while i < n and line[i] in " \t":
        if line[i] == "\t":
            vis = (vis // 8 + 1) * 8
        else:
            vis += 1
        i += 1
    return i, vis


def process_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        bare = raw.rstrip("\r\n")
        m = WITH_RE.match(bare)
        if not m:
            out.append(raw)
            i += 1
            continue

        _, with_vis = visual_indent(bare)
        i += 1

        while i < len(lines) and lines[i].strip() == "":
            out.append(lines[i])
            i += 1
        if i >= len(lines):
            break

        first = lines[i]
        fi, first_vis = visual_indent(first)
        if first_vis <= with_vis:
            out.append(raw)
            continue

        dedent_vis = first_vis - with_vis

        while i < len(lines):
            l = lines[i]
            if not l.strip():
                out.append(l)
                i += 1
                continue
            idx, vis = visual_indent(l)
            if vis <= with_vis:
                break
            new_vis = vis - dedent_vis
            body = l[idx:].rstrip("\r\n")
            if l.endswith("\r\n"):
                eol = "\r\n"
            elif l.endswith("\n"):
                eol = "\n"
            else:
                eol = ""
            out.append(" " * new_vis + body + eol)
            i += 1

    return "".join(out)


def main(paths: list[Path]) -> None:
    for path in paths:
        orig = path.read_text(encoding="utf-8")
        new = process_text(orig)
        if new != orig:
            path.write_text(new, encoding="utf-8")
            print("updated", path)


if __name__ == "__main__":
    roots = [Path(p) for p in sys.argv[1:]]
    py_files: list[Path] = []
    for r in roots:
        if r.is_file() and r.suffix == ".py":
            py_files.append(r)
        elif r.is_dir():
            py_files.extend(sorted(r.rglob("*.py")))
    main(py_files)
