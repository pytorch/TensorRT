# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Fetch a pinned public NVIDIA SegFormer-B0 checkpoint for evaluation."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
REVISION = "21b3847fae21ddee674abd31129307b6a1235bd9"
WEIGHTS_SHA256 = "027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c"


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"model_id": MODEL_ID, "revision": REVISION, "files": {}}
    for name in (
        "config.json",
        "preprocessor_config.json",
        "README.md",
        "pytorch_model.bin",
    ):
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        path = args.output / name
        with urllib.request.urlopen(url, timeout=60) as response, path.open(
            "xb"
        ) as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        digest = checksum(path)
        if name == "pytorch_model.bin" and digest != WEIGHTS_SHA256:
            raise ValueError("Checkpoint SHA256 mismatch")
        record["files"][name] = {
            "url": url,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        print(name, digest, flush=True)
    (args.output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
