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


PI05_COMPONENT_INPUTS = {
    "language": ("inputs_embeds", "attention_mask", "position_ids"),
    "action": (
        "x_t",
        "timestep",
        "prefix_k",
        "prefix_v",
        "position_ids",
        "attention_mask",
    ),
}

PI05_COMPONENT_OUTPUTS = {
    "language": ("lm_hidden_states", "prefix_k", "prefix_v"),
    "action": ("velocity",),
}


def build_pi05_component_artifact(
    engine_dir: str | Path, *, component: str, device_id: int = 0
) -> EdgeExecuTorchArtifact:
    """Embed one PI0.5 component with fixed semantic and engine binding order."""
    if component not in PI05_COMPONENT_INPUTS:
        raise ValueError(f"Unsupported PI0.5 component {component!r}")
    directory = Path(engine_dir)
    try:
        config = json.loads((directory / "config.json").read_text())
        if config["component"] != component:
            raise ValueError(f"Expected {component} component at {directory}")
        if tuple(config["input_names"]) != PI05_COMPONENT_INPUTS[component]:
            raise ValueError(f"Invalid PI0.5 {component} input contract")
        if tuple(config["output_names"]) != PI05_COMPONENT_OUTPUTS[component]:
            raise ValueError(f"Invalid PI0.5 {component} output contract")
        outputs = tuple(
            EdgeOutputSpec.from_dict(
                {"shape": item["shape"], "dtype": _dtype_name(item["dtype"])}
            )
            for item in config["outputs"]
        )
        if len(outputs) != len(PI05_COMPONENT_OUTPUTS[component]):
            raise ValueError(f"Invalid PI0.5 {component} output specifications")
        input_bindings = config.get("trt_input_names", config["input_names"])
        output_bindings = config.get("trt_output_names", config["output_names"])
        if len(input_bindings) != len(PI05_COMPONENT_INPUTS[component]) or len(
            output_bindings
        ) != len(outputs):
            raise ValueError(f"Invalid PI0.5 {component} engine binding counts")
        engine = (directory / config["engine_file"]).read_bytes()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid PI0.5 component at {directory}") from error
    metadata = TensorRTBlobMetadata(
        io_bindings=[
            *[TensorRTIOBinding(name=name, is_input=True) for name in input_bindings],
            *[
                TensorRTIOBinding(
                    name=name,
                    dtype=output.dtype,
                    shape=list(output.shape),
                    is_input=False,
                )
                for name, output in zip(output_bindings, outputs)
            ],
        ],
        device_id=device_id,
    )
    return EdgeExecuTorchArtifact(
        serialize_engine(engine, metadata),
        EdgeComponentMetadata(
            component=component,
            runner=f"pi05_{'prefill' if component == 'language' else 'action'}",
            outputs=outputs,
            runner_config={
                "policy": "pi05",
                "device_id": device_id,
                "input_names": list(PI05_COMPONENT_INPUTS[component]),
                "output_names": list(PI05_COMPONENT_OUTPUTS[component]),
            },
        ).to_json(),
    )


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
            TensorRTIOBinding(
                name=config.get("trt_input_names", input_names)[0], is_input=True
            ),
            TensorRTIOBinding(
                name=config.get("trt_output_names", output_names)[0],
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
            "device_id": device_id,
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
