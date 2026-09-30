"""Compile fixed-shape runtime components through public Torch-TensorRT APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifact import (
    EdgeExecuTorchArtifact,
    build_pi05_component_artifact,
    build_vision_artifact,
)
from .ops import _as_tuple


def compile_component(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    directory: Path,
    *,
    component: str,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    settings: dict[str, Any] | None = None,
) -> EdgeExecuTorchArtifact:
    import tensorrt as trt
    import torch_tensorrt
    from torch_tensorrt.dynamo import CompilationSettings
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    if not args or any(value.device.type != "cuda" for value in args):
        raise ValueError("PI0.5 compilation requires CUDA example inputs")
    device_id = args[0].device.index
    if any(value.device != args[0].device for value in args):
        raise ValueError("Component inputs must use the same CUDA device")
    directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = _as_tuple(module.eval()(*args))
        program = torch.export.export(module, args, strict=False)
        engine_bytes = (
            torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                program,
                arg_inputs=args,
                **{
                    "disable_tf32": True,
                    "use_explicit_typing": True,
                    "truncate_double": True,
                    "timing_cache_path": str(directory / "timing_cache.bin"),
                    **(settings or {}),
                    "device": torch_tensorrt.Device(gpu_id=device_id),
                },
            )
        )
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT did not compile {component}")
    # Export can rename bindings. Preserve the actual names separately from
    # the semantic opset contract, in the same argument/output order.
    with trt.Runtime(trt.Logger(trt.Logger.ERROR)) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Cannot deserialize {component}")
        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            (
                inputs
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else outputs
            ).append(name)
    if len(inputs) != len(input_names) or len(outputs) != len(output_names):
        raise ValueError(f"{component} compiler changed the component I/O contract")
    compiled = TorchTensorRTModule(
        serialized_engine=engine_bytes,
        input_binding_names=inputs,
        output_binding_names=outputs,
        name=f"pi05_{component}",
        settings=CompilationSettings(device=torch_tensorrt.Device(gpu_id=device_id)),
    )
    with torch.no_grad():
        actual = _as_tuple(compiled(*args))
        for expected, result in zip(reference, actual):
            torch.testing.assert_close(result, expected, rtol=2e-2, atol=2e-2)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.engine").write_bytes(engine_bytes)
    config = {
        "model_type": "vit" if component == "vision" else "pi05",
        "component": component,
        "engine_file": "model.engine",
        "input_names": list(input_names),
        "output_names": list(output_names),
        "trt_input_names": inputs,
        "trt_output_names": outputs,
        "outputs": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in actual],
    }
    if component == "vision":
        config["input_layout"] = "hwc"
        config["inputs"] = {input_names[0]: {"shape": list(args[0].shape)}}
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if component == "vision":
        return build_vision_artifact(directory, device_id=device_id)
    return build_pi05_component_artifact(
        directory, component=component, device_id=device_id
    )
