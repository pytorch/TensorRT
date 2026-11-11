import inspect
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from torch_tensorrt_mcp.models import (
    BenchmarkRequest,
    CompilationOptions,
    ModelRequest,
)
from torch_tensorrt_mcp.policy import Policy, PolicyError, confined_path
from torch_tensorrt_mcp.worker import _samples, _verify_provenance


def test_policy_creates_missing_output_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "nested" / "output"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "workspace_root": str(workspace),
                "output_root": str(output),
                "models": {},
                "inputs": {},
            }
        )
    )

    policy = Policy.load(config)

    assert policy.output_root == output
    assert output.is_dir()
    candidate_root = output / ".lowering_pass_candidates"
    assert candidate_root.is_dir()
    assert stat.S_IMODE(candidate_root.stat().st_mode) == 0o700


def test_settings_and_factory_arguments_are_strict(policy):
    request = ModelRequest(
        model="model",
        inputs="inputs",
        settings={
            "truncate_double": True,
            "use_fast_partitioner": False,
            "enable_experimental_decompositions": True,
            "assume_dynamic_shape_support": True,
            "sparse_weights": True,
            "num_avg_timing_iters": 4,
            "max_aux_streams": 2,
            "offload_module_to_cpu": True,
            "decompose_attention": True,
            "fallback_data_dependent_ops": True,
        },
    )
    assert request.settings.num_avg_timing_iters == 4
    assert request.settings.max_aux_streams == 2
    assert request.settings.fallback_data_dependent_ops is True
    assert (
        ModelRequest(model="model", inputs="inputs", seed=2**64 - 1).seed == 2**64 - 1
    )
    with pytest.raises(ValidationError):
        ModelRequest(model="model", inputs="inputs", seed=2**64)
    with pytest.raises(ValidationError):
        ModelRequest(model="model", inputs="inputs", settings={"precision": "fp16"})

    with pytest.raises(ValidationError):
        ModelRequest(
            model="model",
            inputs="inputs",
            settings={"timing_cache_path": "/etc/escape"},
        )
    with pytest.raises(ValidationError):
        ModelRequest(model="model", inputs="inputs", model_parameters={"x": object()})
    with pytest.raises(ValidationError):
        ModelRequest(model="model", inputs="inputs", input_parameters={"seed": 1})
    with pytest.raises(PolicyError):
        policy.validate_request(
            "compile",
            ModelRequest(
                model="model", inputs="inputs", model_parameters={"unexpected": 1}
            ).model_dump(),
        )
    with pytest.raises(ValidationError):
        BenchmarkRequest(source_job_id="a" * 32, iterations=True)
    with pytest.raises(ValidationError):
        BenchmarkRequest(source_job_id="a" * 32, gpu_isolation="isolated")
    for settings in (
        {"num_avg_timing_iters": 0},
        {"num_avg_timing_iters": 101},
        {"max_aux_streams": -1},
        {"max_aux_streams": 1025},
    ):
        with pytest.raises(ValidationError):
            ModelRequest(model="model", inputs="inputs", settings=settings)


def test_compilation_option_names_remain_supported_by_core():
    compiler = pytest.importorskip("torch_tensorrt.dynamo._compiler")
    signature = inspect.signature(compiler.compile)
    core_options = set(signature.parameters)
    adapter_options = set(CompilationOptions.model_fields)
    assert adapter_options <= core_options
    assert CompilationOptions().optimization_level is None
    assert signature.parameters["optimization_level"].default is None


def test_historical_model_request_does_not_reapply_current_timeout_limit(policy):
    historical = ModelRequest(
        model="model",
        inputs="inputs",
        timeout_seconds=policy.max_timeout_seconds + 1,
    ).model_dump()

    with pytest.raises(PolicyError, match="timeout_seconds"):
        policy.validate_request("compile", historical)
    policy.validate_model_request(historical)


def test_input_sample_label_and_provenance_metadata_are_preserved():
    sample = _samples(
        {
            "args": (),
            "sample_id": "sample-1",
            "profile": "opt",
            "label": "representative",
        }
    )[0]
    assert sample.label == "representative"

    original = {
        "weights_sha256": "weights",
        "constants_sha256": "constants",
        "input_sha256": ["input"],
        "graph_sha256": "graph",
        "range_constraints": {},
        "samples": [
            {
                "sample_id": "sample-1",
                "profile": "opt",
                "label": "representative",
            }
        ],
    }
    _verify_provenance(dict(original), original)
    changed = {
        **original,
        "samples": [{**original["samples"][0], "label": "changed"}],
    }
    with pytest.raises(PolicyError, match="samples"):
        _verify_provenance(changed, original)
    with pytest.raises(PolicyError, match="constants_sha256"):
        _verify_provenance(
            {**original, "constants_sha256": "changed"},
            original,
        )


def test_traversal_and_symlinks_are_rejected_without_pinning_factory_source(
    policy, tmp_path
):
    for path in ("../outside", "/etc/passwd", "factories.py/../factories.py"):
        with pytest.raises(PolicyError):
            confined_path(policy.workspace_root, path)
    (policy.workspace_root / "link").symlink_to(policy.workspace_root / "factories.py")
    with pytest.raises(PolicyError):
        confined_path(policy.workspace_root, "link")
    (policy.workspace_root / "factories.py").write_text(
        "def model(**kwargs): return 'changed'\ndef inputs(**kwargs): return ()\n"
    )
    policy.validate_request(
        "compile", ModelRequest(model="model", inputs="inputs").model_dump()
    )
    factory = policy.load_factory(policy.models["model"])
    assert factory() == "changed"


def test_adapter_import_is_gpu_independent():
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, "
            + repr(source)
            + "); import torch_tensorrt_mcp.server; assert 'torch' not in sys.modules; assert 'tensorrt' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_factory_module_supports_dataclasses_and_shared_class_identity(
    policy, tmp_path
):
    (policy.workspace_root / "constants.py").write_text("VALUE = 7\n")
    (policy.workspace_root / "factories.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from constants import VALUE\n"
        "@dataclass\n"
        "class Sample:\n"
        "    value: int = VALUE\n"
        "def model(): return Sample\n"
        "def inputs(seed=0): return Sample(seed)\n"
    )
    # Reload reviewed administrator configuration after changing trusted source.
    reviewed = Policy.load(tmp_path / "config.json")
    model_factory = reviewed.load_factory(reviewed.models["model"])
    input_factory = reviewed.load_factory(reviewed.inputs["inputs"])
    sample_type = model_factory()
    assert sample_type().value == 7
    assert type(input_factory(seed=3)) is sample_type
