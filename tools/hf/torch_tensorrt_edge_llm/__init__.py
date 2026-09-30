"""Standalone Edge-LLM opset built on Torch-TensorRT.

Import this package to register ``torch.ops.tensorrt_edge_llm.*``. Registration
does not require ExecuTorch or model exporters. Native delegate lowering is
available through the public Torch-TensorRT partitioner extension API.
"""

from typing import Any

from . import ops

__all__ = ["ops", "EdgeLLMBackend", "EdgeLLMPartitioner"]


def __getattr__(name: str) -> Any:
    if name in {"EdgeLLMBackend", "EdgeLLMPartitioner"}:
        from . import executorch

        return getattr(executorch, name)
    raise AttributeError(name)
