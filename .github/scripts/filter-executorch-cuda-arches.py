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

# PyTorch accepts named architectures and expands each to a version list before building
# its gencode flags (torch/utils/cpp_extension.py). Expand them the same way here, or a
# name would pass through untouched and carry its sub-6.1 members into the build: Pascal
# alone expands to 6.0;6.1+PTX, and 6.0 is precisely what this filter exists to drop.
#
# Ordered longest-key-first, and applied by substring replacement, because that is what
# torch does: "Maxwell+Tegra" has to be replaced before "Maxwell", or the suffix is left
# behind as "5.0;5.2+PTX+Tegra".
NAMED_ARCHES = (
    ("Kepler+Tesla", "3.7"),
    ("Kepler", "3.5+PTX"),
    ("Maxwell+Tegra", "5.3"),
    ("Maxwell", "5.0;5.2+PTX"),
    ("Pascal", "6.0;6.1+PTX"),
    ("Volta+Tegra", "7.2"),
    ("Volta", "7.0+PTX"),
    ("Turing", "7.5+PTX"),
    ("Ampere+Tegra", "8.7"),
    ("Ampere", "8.0;8.6+PTX"),
    ("Ada", "8.9+PTX"),
    ("Hopper", "9.0+PTX"),
    ("Blackwell+Tegra", "11.0"),
    ("Blackwell", "10.0;10.3;12.0;12.1+PTX"),
    ("Rubin", "10.7+PTX"),
)


def filter_arches(requested: str) -> list[str]:
    for name, versions in NAMED_ARCHES:
        requested = requested.replace(name, versions)
    kept = []
    # A list may be comma, space or semicolon separated. torch normalises spaces the same way
    # (torch/utils/cpp_extension.py does _arch_list.replace(' ', ';')), so accept all three rather
    # than silently passing a space-separated list through unfiltered.
    for entry in requested.replace(",", ";").replace(" ", ";").split(";"):
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
