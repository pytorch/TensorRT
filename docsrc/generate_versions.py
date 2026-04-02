#!/usr/bin/env python3
"""Generates versions.json for the pytorch_sphinx_theme2 version switcher.

Add a new entry here whenever a new release is published.
The file is written to ../docs/versions.json by `make html`.
"""

import json
import sys

VERSIONS = [
    {
        "name": "main (latest)",
        "version": "main",
        "url": "https://pytorch.org/TensorRT/",
        "preferred": True,
    },
    {
        "name": "v2.10.0",
        "version": "v2.10.0",
        "url": "https://pytorch.org/TensorRT/v2.10.0/",
    },
    {
        "name": "v2.9.0",
        "version": "v2.9.0",
        "url": "https://pytorch.org/TensorRT/v2.9.0/",
    },
    {
        "name": "v2.8.0",
        "version": "v2.8.0",
        "url": "https://pytorch.org/TensorRT/v2.8.0/",
    },
    {
        "name": "v2.7.0",
        "version": "v2.7.0",
        "url": "https://pytorch.org/TensorRT/v2.7.0/",
    },
    {
        "name": "v2.6.1",
        "version": "v2.6.1",
        "url": "https://pytorch.org/TensorRT/v2.6.1/",
    },
    {
        "name": "v2.6.0",
        "version": "v2.6.0",
        "url": "https://pytorch.org/TensorRT/v2.6.0/",
    },
    {
        "name": "v2.5.0",
        "version": "v2.5.0",
        "url": "https://pytorch.org/TensorRT/v2.5.0/",
    },
    {
        "name": "v2.4.0",
        "version": "v2.4.0",
        "url": "https://pytorch.org/TensorRT/v2.4.0/",
    },
    {
        "name": "v2.3.0",
        "version": "v2.3.0",
        "url": "https://pytorch.org/TensorRT/v2.3.0/",
    },
    {
        "name": "v2.2.0",
        "version": "v2.2.0",
        "url": "https://pytorch.org/TensorRT/v2.2.0/",
    },
    {
        "name": "v2.1.0",
        "version": "v2.1.0",
        "url": "https://pytorch.org/TensorRT/v2.1.0/",
    },
    {
        "name": "v1.4.0",
        "version": "v1.4.0",
        "url": "https://pytorch.org/TensorRT/v1.4.0/",
    },
    {
        "name": "v1.3.0",
        "version": "v1.3.0",
        "url": "https://pytorch.org/TensorRT/v1.3.0/",
    },
    {
        "name": "v1.2.0",
        "version": "v1.2.0",
        "url": "https://pytorch.org/TensorRT/v1.2.0/",
    },
    {
        "name": "v1.1.1",
        "version": "v1.1.1",
        "url": "https://nvidia.github.io/Torch-TensorRT/v1.1.1/",
    },
    {
        "name": "v1.1.0",
        "version": "v1.1.0",
        "url": "https://nvidia.github.io/Torch-TensorRT/v1.1.0/",
    },
    {
        "name": "v1.0.0",
        "version": "v1.0.0",
        "url": "https://nvidia.github.io/Torch-TensorRT/v1.0.0/",
    },
    {
        "name": "v0.4.1",
        "version": "v0.4.1",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.4.1/",
    },
    {
        "name": "v0.4.0",
        "version": "v0.4.0",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.4.0/",
    },
    {
        "name": "v0.3.0",
        "version": "v0.3.0",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.3.0/",
    },
    {
        "name": "v0.2.0",
        "version": "v0.2.0",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.2.0/",
    },
    {
        "name": "v0.1.0",
        "version": "v0.1.0",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.1.0/",
    },
    {
        "name": "v0.0.3",
        "version": "v0.0.3",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.0.3/",
    },
    {
        "name": "v0.0.2",
        "version": "v0.0.2",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.0.2/",
    },
    {
        "name": "v0.0.1",
        "version": "v0.0.1",
        "url": "https://nvidia.github.io/Torch-TensorRT/v0.0.1/",
    },
]

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    payload = json.dumps(VERSIONS, indent=2)
    if out:
        with open(out, "w") as f:
            f.write(payload + "\n")
        print(f"Written {out}")
    else:
        print(payload)
