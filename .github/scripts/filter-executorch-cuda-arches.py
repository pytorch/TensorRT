#!/usr/bin/env python3
"""Drop CUDA architectures ExecuTorch's shims cannot compile for.

The shims call ``__dp4a``, which nvcc does not declare before sm_61, so a list carrying
5.0 or 6.0 fails with ``identifier "__dp4a" is undefined`` whichever CUDA is in use. The
cu126 build environment asks for ``5.0;6.0;7.0;7.5;8.0;8.6;9.0`` while the CUDA 13 ones
start at 7.5, which is why only cu126 broke.

Filtering what the environment asked for rather than pinning a list here, so a channel
that adds a newer architecture still builds for it without editing this file.
"""

from __future__ import annotations

import sys

# The first architecture that declares __dp4a. GP100 is 6.0 and does not have it; the
# rest of Pascal is 6.1 and does.
MINIMUM = 6.1


def filter_arches(requested: str) -> list[str]:
    kept = []
    for entry in requested.replace(",", ";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        # An entry may carry a suffix such as "8.6+PTX", which sorts by its number.
        number = entry.split("+", 1)[0]
        try:
            if float(number) >= MINIMUM:
                kept.append(entry)
        except ValueError:
            # Not a version this understands. Keep it and let the compiler judge,
            # rather than silently dropping a target that may be real.
            kept.append(entry)
    return kept


def main(argv: list[str]) -> int:
    requested = argv[1] if len(argv) > 1 else ""
    kept = filter_arches(requested)
    if requested.strip() and not kept:
        # Printing nothing would leave nvcc to pick its own default, which is not what
        # the channel asked for and would ship a wheel for the wrong architectures.
        print(
            f"No CUDA architecture in {requested!r} is at or above sm_61, "
            "which ExecuTorch's CUDA shims require.",
            file=sys.stderr,
        )
        return 1
    print(";".join(kept))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
