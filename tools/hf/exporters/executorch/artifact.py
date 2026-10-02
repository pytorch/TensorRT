from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch_tensorrt.executorch.serialization import (
    TensorRTBlobMetadata,
    TensorRTIOBinding,
    serialize_engine,
)

from .serialization import EdgeComponentMetadata, EdgeOutputSpec


@dataclass(frozen=True)
class EdgeExecuTorchArtifact:
    trt_blob: bytes
    edge_metadata_json: str


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("torch.")


def build_vision_artifact(
    engine_dir: str | Path,
    *,
    device_id: int = 0,
) -> EdgeExecuTorchArtifact:
    """Build the nested TR01 + Edge metadata for one VitRunner component."""
    component_dir = Path(engine_dir)
    config_path = component_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
        if config["component"] != "vision" or config["model_type"] != "vit":
            raise ValueError(
                "Edge vision artifact requires component='vision' and model_type='vit'"
            )
        input_names = list(config["input_names"])
        input_config = dict(
            config.get("inputs", {}).get(input_names[0], {}) if input_names else {}
        )
        input_shape = list(input_config.get("shape", []))
        input_layout = config.get("input_layout")
        if input_layout is None and len(input_shape) == 4 and input_shape[-1] == 3:
            input_layout = "hwc"
        if input_layout != "hwc":
            raise ValueError("Edge VitRunner artifact requires input_layout='hwc'")
        output_names = list(config["output_names"])
        outputs = list(config["outputs"])
        engine_file = str(config["engine_file"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid Edge vision config at {config_path}") from exc

    if len(input_names) != 1 or len(output_names) != 1 or len(outputs) != 1:
        raise ValueError(
            "The first Edge vision delegate requires exactly one input and one output"
        )

    engine_path = component_dir / engine_file
    try:
        engine_bytes = engine_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"Vision engine was not found at {engine_path}") from exc

    output_spec = EdgeOutputSpec.from_dict(
        {
            "shape": outputs[0]["shape"],
            "dtype": _dtype_name(outputs[0]["dtype"]),
        }
    )
    trt_metadata = TensorRTBlobMetadata(
        io_bindings=[
            TensorRTIOBinding(name=input_names[0], is_input=True),
            TensorRTIOBinding(
                name=output_names[0],
                dtype=output_spec.dtype,
                shape=list(output_spec.shape),
                is_input=False,
            ),
        ],
        device_id=device_id,
    )
    edge_metadata = EdgeComponentMetadata(
        component="vision",
        runner="vit",
        outputs=(output_spec,),
        runner_config={
            "model_type": "vit",
            "input_layout": "hwc",
            "input_dtype": _dtype_name(
                config.get("input_dtype", input_config.get("dtype", "float16"))
            ),
        },
    )
    return EdgeExecuTorchArtifact(
        trt_blob=serialize_engine(engine_bytes, trt_metadata),
        edge_metadata_json=edge_metadata.to_json(),
    )


def build_language_artifact(
    engine_dir: str | Path,
    *,
    device_id: int = 0,
) -> EdgeExecuTorchArtifact:
    component_dir = Path(engine_dir)
    config_path = component_dir / "config.json"

    try:
        config = json.loads(config_path.read_text())
        if config["component"] != "language" or config["model_type"] != "language":
            raise ValueError(
                "Language artifact requires component='language' "
                "and model_type='language'"
            )

        input_names = list(config["input_names"])
        output_names = list(config["output_names"])
        outputs = list(config["outputs"])
        engine_file = str(config["engine_file"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid Edge language config at {config_path}") from exc

    if not input_names:
        raise ValueError("Language artifact requires at least one input")

    if len(output_names) != len(outputs):
        raise ValueError("Language output names and output specifications must match")

    engine_path = component_dir / engine_file
    engine_bytes = engine_path.read_bytes()

    output_specs = tuple(
        EdgeOutputSpec.from_dict(
            {
                "shape": output["shape"],
                "dtype": _dtype_name(output["dtype"]),
            }
        )
        for output in outputs
    )

    bindings = [TensorRTIOBinding(name=name, is_input=True) for name in input_names]
    bindings.extend(
        TensorRTIOBinding(
            name=name,
            dtype=spec.dtype,
            shape=list(spec.shape),
            is_input=False,
        )
        for name, spec in zip(output_names, output_specs)
    )

    trt_metadata = TensorRTBlobMetadata(
        io_bindings=bindings,
        device_id=device_id,
    )
    edge_metadata = EdgeComponentMetadata(
        component="language",
        runner="llm_prefill",
        outputs=output_specs,
        runner_config={
            "model_type": "language",
            "context_attention_mask_type": config.get("context_attention_mask_type"),
            "prefix_pad_mask_len": config.get("prefix_pad_mask_len"),
        },
    )

    return EdgeExecuTorchArtifact(
        trt_blob=serialize_engine(engine_bytes, trt_metadata),
        edge_metadata_json=edge_metadata.to_json(),
    )
