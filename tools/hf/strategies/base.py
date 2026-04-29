"""
Abstract base class for per-family compilation + benchmarking strategies.

Each strategy owns:
  load()      – download & prepare the model (called once, on CPU or GPU)
  compile()   – export + TRT compile (or torch.compile fast path)
  benchmark() – warmup + timed loop + return list[dict] rows for metrics.py
  generate()  – optional: run a single sample-mode forward (text, image, audio)
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Literal, Optional

# Type aliases used in both RunConfig and CLIArgs.
Precision = Literal["FP16", "BF16", "FP32"]
Mode = Literal["export", "compile"]
KVCache = Literal["static_v1", "static_v2", "hf_static"]
EngineFormat = Literal["exported_program", "torchscript", "aot_inductor"]


@dataclasses.dataclass
class RunConfig:
    """Compile and benchmark any HuggingFace model with Torch-TensorRT."""

    model: str
    """HuggingFace model tag or local path (e.g. 'bert-base-uncased')."""

    task: Optional[str] = None
    """HF pipeline task to override family auto-detection (e.g. 'text-generation')."""

    precision: Precision = "FP16"
    """Target compute precision."""

    autocast: bool = False
    """Use TRT autocast (Path B): model stays FP32, TRT compiler casts."""

    mode: Mode = "export"
    """`export` = torch.export + dynamo.compile; `compile` = torch.compile fast path."""

    batch_size: int = 1
    """Batch size for compilation and benchmarking."""

    iterations: int = 10
    """Number of timed iterations during benchmarking."""

    min_block_size: int = 1
    """Minimum consecutive TRT-compatible ops to form a TRT segment."""

    offload_module_to_cpu: bool = False
    """Offload original PyTorch module to CPU after compile (memory-constrained models)."""

    engine_cache_dir: Optional[str] = None
    """Directory to persist built engines across runs."""

    debug: bool = False
    """Enable Torch-TensorRT debug logging."""

    save_engine: Optional[str] = None
    """Path to save a torch_tensorrt module (Python wrapper around the engine).
    Output format is controlled by --engine-format."""

    engine_format: EngineFormat = "exported_program"
    """Serialization format for --save-engine.
      exported_program : torch.export.ExportedProgram with embedded TRT engine (.ep)
      torchscript      : TorchScript module containing the TRT engine (.ts)
      aot_inductor     : AOT Inductor package (Linux only)"""

    save_trt_engine: Optional[str] = None
    """Path to save a raw serialized TensorRT engine (bytes from buildSerializedNetwork).
    Loadable by the standalone TensorRT runtime (TRT-LLM, Triton, C++ TRT API).
    Requires a fully TRT-compatible graph (no PyTorch fallback partitions)."""

    save_exported_program: Optional[str] = None
    """Path to save the pre-TRT torch.export.ExportedProgram (no compilation).
    Useful for inspection, archival, or feeding into a different backend."""

    # ---- LLM-specific ----
    isl: int = 128
    """Input sequence length (LLM)."""

    num_tokens: int = 64
    """New tokens to allow at compile time; max_seq_len = isl + num_tokens (LLM)."""

    cache: Optional[KVCache] = None
    """KV cache strategy (LLM)."""

    prompt: str = "What is parallel programming?"
    """Prompt for qualitative LLM output (used with --generate)."""

    # ---- Diffusion-specific ----
    image_size: int = 512
    """Output image size H=W (diffusion)."""

    num_inference_steps: int = 20
    """Number of denoising steps (diffusion / video diffusion)."""

    num_frames: int = 16
    """Number of video frames to generate (video diffusion)."""

    # ---- Audio-specific ----
    audio_duration_s: float = 30.0
    """Simulated audio duration in seconds for RTF calculation (audio)."""

    # ---- Accuracy comparison ----
    accuracy_atol: float = 1e-2
    """Absolute tolerance for accuracy --check; tuned for FP16."""

    accuracy_rtol: float = 1e-2
    """Relative tolerance for accuracy --check."""

    accuracy_cos_sim_min: float = 0.99
    """Minimum cosine similarity required for the accuracy check to pass."""


@dataclasses.dataclass
class CLIArgs(RunConfig):
    """Top-level CLI args = RunConfig + run-time output flags."""

    benchmark: bool = False
    """Run the benchmarking loop and report metrics."""

    generate: bool = False
    """Run a single forward pass and print qualitative output."""

    accuracy: bool = False
    """Compare PyTorch and TRT output tensors and report cos_sim / abs_err / allclose."""

    json_out: Optional[str] = None
    """Write benchmark results to this JSON file."""


class ModelStrategy(abc.ABC):
    """Base class for a model-family compilation + benchmarking strategy."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg

    @abc.abstractmethod
    def load(self) -> None:
        """Load the model weights from HF hub onto CPU/GPU."""

    @abc.abstractmethod
    def compile(self) -> None:
        """Export and TRT-compile (or torch.compile) the loaded model."""

    @abc.abstractmethod
    def benchmark(self) -> list[dict]:
        """Run the benchmarking loop and return result-row dicts."""

    def generate(self) -> None:
        """Optional: run a single forward pass and print qualitative output."""

    def accuracy(self) -> list[dict]:
        """
        Compare PyTorch and TRT outputs on the same inputs.  Default
        implementation calls _run_pt() / _run_trt() (each strategy must
        provide them or override this method).
        """
        from common.accuracy import compare_outputs, print_accuracy_table

        if not hasattr(self, "_run_pt") or not hasattr(self, "_run_trt"):
            print(
                f"[{type(self).__name__}] accuracy() not implemented for this "
                f"strategy (no _run_pt / _run_trt hooks)."
            )
            return []

        pt_out = self._run_pt()  # type: ignore[attr-defined]
        trt_out = self._run_trt()  # type: ignore[attr-defined]

        rows = compare_outputs(
            pt_out,
            trt_out,
            atol=self.cfg.accuracy_atol,
            rtol=self.cfg.accuracy_rtol,
        )
        print_accuracy_table(
            rows,
            title=f"Accuracy – {self.cfg.model}",
            cos_sim_min=self.cfg.accuracy_cos_sim_min,
        )
        return rows
