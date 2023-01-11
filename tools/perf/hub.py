import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from transformers import BertModel, BertTokenizer, BertConfig
import os
import json
import custom_models as cm

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
VALID_PATHS = ("script", "trace", "torchscript", "pytorch", "all")

# Key models selected for benchmarking with their respective paths
BENCHMARK_MODELS = {
    "vgg16": {
        "model": models.vgg16(weights=models.VGG16_Weights.DEFAULT),
        "path": ["script", "pytorch"],
    },
    "resnet50": {
        "model": models.resnet50(weights=None),
        "path": ["script", "pytorch"],
    },
    "efficientnet_b0": {
        "model": timm.create_model("efficientnet_b0", pretrained=True),
        "path": ["script", "pytorch"],
    },
    "vit": {
        "model": timm.create_model("vit_base_patch16_224", pretrained=True),
        "path": "script",
    },
    "bert_base_uncased": {"model": cm.BertModule(), "path": "trace"},
}


def get(n, m, manifest):
    print("Downloading {}".format(n))
    traced_filename = "models/" + n + "_traced.jit.pt"
    script_filename = "models/" + n + "_scripted.jit.pt"
    pytorch_filename = "models/" + n + "_pytorch.pt"
    x = torch.ones((1, 3, 300, 300)).cuda()
    if n == "bert_base_uncased":
        traced_model = m["model"]
        torch.jit.save(traced_model, traced_filename)
        manifest.update({n: [traced_filename]})
    else:
        m["model"] = m["model"].eval().cuda()

        # Get all desired model save specifications as list
        paths = [m["path"]] if isinstance(m["path"], str) else m["path"]

        # Depending on specified model save specifications, save desired model formats
        if any(path in ("all", "torchscript", "trace") for path in paths):
            # (TorchScript) Traced model
            trace_model = torch.jit.trace(m["model"], [x])
            torch.jit.save(trace_model, traced_filename)
            manifest.update({n: [traced_filename]})
        if any(path in ("all", "torchscript", "script") for path in paths):
            # (TorchScript) Scripted model
            script_model = torch.jit.script(m["model"])
            torch.jit.save(script_model, script_filename)
            if n in manifest.keys():
                files = list(manifest[n]) if type(manifest[n]) != list else manifest[n]
                files.append(script_filename)
                manifest.update({n: files})
            else:
                manifest.update({n: [script_filename]})
        if any(path in ("all", "pytorch") for path in paths):
            # (PyTorch Module) model
            torch.save(m["model"], pytorch_filename)
            if n in manifest.keys():
                files = list(manifest[n]) if type(manifest[n]) != list else manifest[n]
                files.append(script_filename)
                manifest.update({n: files})
            else:
                manifest.update({n: [script_filename]})
    return manifest


def download_models(version_matches, manifest):
    # Download all models if torch version is different than model version
    if not version_matches:
        for n, m in BENCHMARK_MODELS.items():
            manifest = get(n, m, manifest)
    else:
        for n, m in BENCHMARK_MODELS.items():
            scripted_filename = "models/" + n + "_scripted.jit.pt"
            traced_filename = "models/" + n + "_traced.jit.pt"
            pytorch_filename = "models/" + n + "_pytorch.pt"
            # Check if model file exists on disk

            # Extract model specifications as list and ensure all desired formats exist
            paths = [m["path"]] if isinstance(m["path"], str) else m["path"]
            if (
                (
                    any(path == "all" for path in paths)
                    and os.path.exists(scripted_filename)
                    and os.path.exists(traced_filename)
                    and os.path.exists(pytorch_filename)
                )
                or (
                    any(path == "torchscript" for path in paths)
                    and os.path.exists(scripted_filename)
                    and os.path.exists(traced_filename)
                )
                or (
                    any(path == "script" for path in paths)
                    and os.path.exists(scripted_filename)
                )
                or (
                    any(path == "trace" for path in paths)
                    and os.path.exists(traced_filename)
                )
                or (
                    any(path == "pytorch" for path in paths)
                    and os.path.exists(pytorch_filename)
                )
            ):
                print("Skipping {} ".format(n))
                continue
            manifest = get(n, m, manifest)


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
                print(
                    "Torch version: {} mismatches \
                with manifest's version: {}. Re-downloading \
                all models".format(
                        torch_version, manifest["version"]
                    )
                )

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
    # Ensure all specified desired model formats exist and are valid
    paths = [
        [m["path"]] if isinstance(m["path"], str) else m["path"]
        for m in BENCHMARK_MODELS.values()
    ]
    assert all(
        (path in VALID_PATHS) for path_list in paths for path in path_list
    ), "Not all 'path' attributes in BENCHMARK_MODELS are valid"
    main()
