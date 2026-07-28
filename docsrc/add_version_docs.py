#!/usr/bin/env python3
"""Add pre-built docs for a new release tag to the docs/ tree.

After tagging a release on main, run:
    python3 docsrc/add_version_docs.py v2.11.0

This will:
  1. Extract the pre-built docs from that git tag into docs/v2.11.0/
  2. Insert the version entry into docsrc/generate_versions.py (descending order)
  3. Regenerate docs/versions.json
  4. Stage all changes with git add
"""

import re
import subprocess
import sys
from pathlib import Path


def ver_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.lstrip("v").split("."))


def extract_docs(repo_root: Path, version: str) -> None:
    target = repo_root / "docs" / version
    if target.exists():
        print(f"Error: docs/{version} already exists", file=sys.stderr)
        sys.exit(1)
    target.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", version, "--", "docs/"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["tar", "-x", "--strip-components=1", "-C", str(target)],
        input=archive.stdout,
        check=True,
    )
    count = sum(1 for _ in target.rglob("*"))
    print(f"  Extracted {count} files to docs/{version}/")


def insert_into_generate_versions(gen_versions: Path, version: str) -> None:
    src = gen_versions.read_text()

    # Sanity check: don't add duplicates
    if f'"version": "{version}"' in src:
        print(f"  {version} already present in generate_versions.py, skipping")
        return

    new_entry = (
        f"    {{\n"
        f'        "name": "{version}",\n'
        f'        "version": "{version}",\n'
        f'        "url": "https://pytorch.org/TensorRT/{version}/",\n'
        f"    }},\n"
    )

    new_ver = ver_tuple(version)

    # Find the first existing vX.Y.Z entry whose version is strictly less than
    # the new one — insert before it to maintain descending order.
    insert_pos = None
    for m in re.finditer(r'    \{\n        "name": "(v[\d.]+)"', src):
        if ver_tuple(m.group(1)) < new_ver:
            insert_pos = m.start()
            break

    if insert_pos is None:
        # All existing entries are >= new version; append before closing bracket
        insert_pos = src.rfind("]")
        if insert_pos == -1:
            print(
                "Error: could not find insertion point in generate_versions.py",
                file=sys.stderr,
            )
            sys.exit(1)

    gen_versions.write_text(src[:insert_pos] + new_entry + src[insert_pos:])
    print(f"  Inserted {version} into generate_versions.py")


def regenerate_versions_json(docsrc: Path) -> None:
    subprocess.run(
        [sys.executable, "generate_versions.py", "../docs/versions.json"],
        cwd=docsrc,
        check=True,
    )
    print("  Regenerated docs/versions.json")


def stage(repo_root: Path, version: str, docsrc: Path) -> None:
    subprocess.run(
        [
            "git",
            "add",
            f"docs/{version}",
            "docs/versions.json",
            str(docsrc / "generate_versions.py"),
        ],
        cwd=repo_root,
        check=True,
    )
    print("  Staged changes")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <version-tag>  (e.g. v2.11.0)", file=sys.stderr)
        sys.exit(1)

    version = sys.argv[1]
    repo_root = Path(__file__).parent.parent.resolve()
    docsrc = repo_root / "docsrc"

    # Validate the tag exists in git history
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{version}"],
        cwd=repo_root,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"Error: tag '{version}' not found. Did you fetch tags?", file=sys.stderr)
        sys.exit(1)

    print(f"Adding docs for {version}...")
    extract_docs(repo_root, version)
    insert_into_generate_versions(docsrc / "generate_versions.py", version)
    regenerate_versions_json(docsrc)
    stage(repo_root, version, docsrc)
    print(f"\nDone. Review with: git diff --cached --stat")
    print(f"Then commit:        git commit -m 'docs: add versioned docs for {version}'")


if __name__ == "__main__":
    main()
