# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import torch

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

torch_version = torch.__version__

# Detect case of no GPU before deserialization of models on GPU
if not torch.cuda.is_available():
    raise Exception(
        "No GPU found. Please check if installed torch version is compatible with CUDA version"
    )

# Downloads all model files again if manifest file is not present
MANIFEST_FILE = "model_manifest.json"

# Valid paths for model-saving specification
VALID_PATHS = ("export", "pytorch", "all")

# Key models selected for benchmarking with their respective paths
from utils import BENCHMARK_MODELS


def get(n, m, manifest):
    print("Downloading {}".format(n))
    exported_filename = "models/" + n + "_exported.pt2"
    pytorch_filename = "models/" + n + "_pytorch.pt"
    model = m["model"].eval().cuda()
    paths = [m["path"]] if isinstance(m["path"], str) else m["path"]
    files = []
    if any(path in ("all", "export") for path in paths):
        inputs = tuple(inp.cuda() for inp in m["inputs"])
        torch.export.save(torch.export.export(model, inputs), exported_filename)
        files.append(exported_filename)
    if any(path in ("all", "pytorch") for path in paths):
        torch.save(model, pytorch_filename)
        files.append(pytorch_filename)
    manifest[n] = files

    return manifest


def download_models(version_matches, manifest):
    for n, m in BENCHMARK_MODELS.items():
        paths = [m["path"]] if isinstance(m["path"], str) else m["path"]
        if not all(path in VALID_PATHS for path in paths):
            raise ValueError("Unsupported benchmark model format")
        files = []
        if any(path in ("all", "export") for path in paths):
            files.append("models/" + n + "_exported.pt2")
        if any(path in ("all", "pytorch") for path in paths):
            files.append("models/" + n + "_pytorch.pt")
        if (
            version_matches
            and manifest.get(n) == files
            and all(os.path.exists(path) for path in files)
        ):
            print("Skipping {} ".format(n))
            continue
        get(n, m, manifest)


def main():
    manifest = None
    version_matches = False

    # Check if Manifest file exists or is empty
    if not os.path.exists(MANIFEST_FILE) or os.stat(MANIFEST_FILE).st_size == 0:
        manifest = {"version": torch_version}

        # Creating an empty manifest file for overwriting post setup
        os.system("touch {}".format(MANIFEST_FILE))
    else:
        # Load manifest if already exists
        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
            if manifest["version"] == torch_version:
                version_matches = True
            else:
                print("Torch version: {} mismatches \
                with manifest's version: {}. Re-downloading \
                all models".format(torch_version, manifest["version"]))

                # Overwrite the manifest version as current torch version
                manifest["version"] = torch_version

    download_models(version_matches, manifest)

    # Write updated manifest file to disk
    with open(MANIFEST_FILE, "r+") as f:
        data = f.read()
        f.seek(0)
        record = json.dumps(manifest)
        f.write(record)
        f.truncate()


if __name__ == "__main__":
    main()
