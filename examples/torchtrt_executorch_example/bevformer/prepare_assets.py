# SPDX-License-Identifier: Apache-2.0
"""Download hash-pinned reference sources and optionally the official checkpoint."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


def verified_download(url, destination, expected):
    if destination.exists():
        with destination.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(
                f"Existing file has unexpected contents; not overwriting: {destination}"
            )
        print("Verified:", destination, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded checksum mismatch: {url}")
        os.replace(temporary, destination)
        print("Downloaded and verified:", destination, flush=True)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Also download the approximately 383 MB model checkpoint",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sources = json.loads((here / "sources.json").read_text())
    references = json.loads((here / "reference_sources.json").read_text())
    versions = references["versions"]
    bases = {
        "BEVFormer": f"https://raw.githubusercontent.com/fundamentalvision/BEVFormer/{versions['bevformer']}/",
        "mmdet": f"https://raw.githubusercontent.com/open-mmlab/mmdetection/{versions['mmdetection']}/",
        "mmcv": f"https://raw.githubusercontent.com/open-mmlab/mmcv/{versions['mmcv']}/",
        "torchvision": f"https://raw.githubusercontent.com/pytorch/vision/{versions['torchvision']}/torchvision/transforms/",
    }
    remap = {
        "resnet.py": "mmdet/models/backbones/resnet.py",
        "fpn.py": "mmdet/models/necks/fpn.py",
        "positional_encoding.py": "mmdet/models/utils/positional_encoding.py",
        "transformer.py": "mmcv/cnn/bricks/transformer.py",
        "multi_scale_deform_attn.py": "mmcv/ops/multi_scale_deform_attn.py",
    }
    for relative, checksum in references["sha256"].items():
        path = Path(relative)
        project = path.parts[1]
        remote = (
            remap.get(path.name, path.name)
            if project in ("mmdet", "mmcv")
            else "/".join(path.parts[2:])
        )
        verified_download(bases[project] + remote, args.root / relative, checksum)
    # Component validation also reads provenance from the chosen data root.
    provenance = args.root / "sources.json"
    if provenance.resolve() != (here / "sources.json").resolve():
        if provenance.exists() and json.loads(provenance.read_text()) != sources:
            raise ValueError(f"Different provenance already exists: {provenance}")
        provenance.write_text(json.dumps(sources, indent=2) + "\n")
    if args.checkpoint:
        verified_download(
            sources["checkpoint_url"],
            args.root / "checkpoints/bevformer_tiny_epoch_24.pth",
            sources["checkpoint_sha256"],
        )


if __name__ == "__main__":
    main()
