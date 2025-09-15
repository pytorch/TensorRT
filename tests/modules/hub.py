import importlib
import importlib.util
import json
import os

import custom_models as cm
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

to_test_models = {
    "pooling": {"model": cm.Pool(), "path": "trace"},
    "module_fallback": {"model": cm.ModuleFallbackMain(), "path": "script"},
    "loop_fallback_eval": {"model": cm.LoopFallbackEval(), "path": "script"},
    "loop_fallback_no_eval": {"model": cm.LoopFallbackNoEval(), "path": "script"},
    "conditional": {"model": cm.FallbackIf(), "path": "script"},
    "inplace_op_if": {"model": cm.FallbackInplaceOPIf(), "path": "script"},
    "standard_tensor_input": {"model": cm.StandardTensorInput(), "path": "script"},
    "tuple_input": {"model": cm.TupleInput(), "path": "script"},
    "list_input": {"model": cm.ListInput(), "path": "script"},
    "tuple_input_output": {"model": cm.TupleInputOutput(), "path": "script"},
    "list_input_output": {"model": cm.ListInputOutput(), "path": "script"},
    "list_input_tuple_output": {
        "model": cm.ListInputTupleOutput(),
        "path": "script",
    },
    # "bert_base_uncased": {"model": cm.BertModule(), "path": "trace"},
}


def get(n, m, manifest):
    print("Downloading {}".format(n))
    traced_filename = n + "_traced.jit.pt"
    script_filename = n + "_scripted.jit.pt"
    x = torch.ones((1, 3, 300, 300)).cuda()
    if n == "bert-base-uncased":
        traced_model = m["model"]
        torch.jit.save(traced_model, traced_filename)
        manifest.update({n: [traced_filename]})
    else:
        m["model"] = m["model"].eval().cuda()
        if m["path"] == "both" or m["path"] == "trace":
            trace_model = torch.jit.trace(m["model"], [x])
            torch.jit.save(trace_model, traced_filename)
            manifest.update({n: [traced_filename]})
        if m["path"] == "both" or m["path"] == "script":
            script_model = torch.jit.script(m["model"])
            torch.jit.save(script_model, script_filename)
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
        for n, m in to_test_models.items():
            manifest = get(n, m, manifest)
    else:
        for n, m in to_test_models.items():
            scripted_filename = n + "_scripted.jit.pt"
            traced_filename = n + "_traced.jit.pt"
            # Check if model file exists on disk
            if (
                (
                    m["path"] == "both"
                    and os.path.exists(scripted_filename)
                    and os.path.exists(traced_filename)
                )
                or (m["path"] == "script" and os.path.exists(scripted_filename))
                or (m["path"] == "trace" and os.path.exists(traced_filename))
            ):
                print("Skipping {} ".format(n))
                continue
            manifest = get(n, m, manifest)


def main():

    manifest = None
    version_matches = False
    manifest_exists = False

    # Check if Manifest file exists or is empty
    if not os.path.exists(MANIFEST_FILE) or os.stat(MANIFEST_FILE).st_size == 0:
        manifest = {"version": torch_version}

        # Creating an empty manifest file for overwriting post setup
        os.system("touch {}".format(MANIFEST_FILE))
    else:
        manifest_exists = True

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
    main()
