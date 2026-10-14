"""Check the existing runner's shape and eight printed values, not full output."""

import argparse
import json
import math
import re
from pathlib import Path


def check(report, log):
    shape = re.search(r"output\[0\] shape=\[([^]]+)\]", log)
    if shape is None:
        raise AssertionError("No output tensor in native log")
    actual_shape = [int(dim) for dim in shape.group(1).split(",")]
    if actual_shape != report["output_shape"]:
        raise AssertionError(f"Unexpected shape: {actual_shape}")
    values = re.search(r"first 8 values:([^\n]+)", log)
    if values is None:
        raise AssertionError("No printed values in native log")
    actual = [float(value) for value in values.group(1).split()]
    expected = report["native_ones_first_eight"]
    if len(actual) != len(expected):
        raise AssertionError("Unexpected output value count")
    for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
        # Runner prints only four digits after the decimal point.
        if not math.isfinite(value) or not math.isclose(
            value, reference, abs_tol=6e-5, rel_tol=0
        ):
            raise AssertionError(f"Native sample {index}: {value} != {reference}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    check(json.loads(args.validation.read_text()), args.log.read_text())
    print("PASS: native shape and eight printed values; not a full-output comparison")


if __name__ == "__main__":
    main()
