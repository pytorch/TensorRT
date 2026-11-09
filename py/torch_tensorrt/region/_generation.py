# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for compile-time, agent-authored region kernels."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import torch

#: The ``module:attribute`` form of an extension ``provider`` or ``backend``.
IMPORT_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", re.ASCII)
_PROVIDERS = ("claude", "codex")
_BACKENDS = ("triton", "cutile", "cuda_cpp", "cutedsl")


def split_import_path(path: str) -> tuple[str, str]:
    """Split a ``module:attribute`` extension path, rejecting any other form."""
    if not isinstance(path, str) or IMPORT_PATH.fullmatch(path) is None:
        raise ValueError(f"Expected a 'module:attribute' import path; got {path!r}")
    module, _, attribute = path.partition(":")
    return module, attribute


def _check(
    owner: object,
    name: str,
    kind: type,
    *,
    positive: bool = True,
    optional: bool = False,
) -> None:
    """Raise TypeError for a field of the wrong type, ValueError for a bad number.

    Numbers must be finite and positive, or nonnegative when ``positive=False``.
    An int given for a float field is stored as a float: settings feed the
    task's cache key as JSON, where ``2`` and ``2.0`` differ, and the replay
    CLI reads them back as floats.
    """
    value = getattr(owner, name)
    if optional and value is None:
        return
    sign = "positive" if positive else "nonnegative"
    expected = {
        bool: "a bool",
        str: "a string",
        int: f"a {sign} integer",
        float: f"a finite {sign} number",
    }[kind]
    message = (
        f"{type(owner).__name__} {name} must be {expected}"
        f"{' or None' if optional else ''}; got {value!r}"
    )
    numeric = kind in (int, float)
    # bool subclasses int, but True is neither a count nor a duration.
    if not isinstance(value, (int, float) if kind is float else kind) or (
        numeric and isinstance(value, bool)
    ):
        raise TypeError(message)
    if kind is float and not isinstance(value, float):
        try:
            value = float(value)
        except OverflowError:
            raise ValueError(message) from None
        # The dataclasses are frozen.
        object.__setattr__(owner, name, value)
    if numeric and (not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(message)


def _snapshot_tuning(tuning: Any) -> dict[str, tuple[Any, ...]]:
    """Copy tuning choices as finite JSON; tuples are accepted as arrays."""
    if not isinstance(tuning, dict):
        raise TypeError(f"Tuning must be a dict of knob choices; got {tuning!r}")
    snapshot = {}
    for name, choices in tuning.items():
        if not isinstance(name, str) or not isinstance(choices, (list, tuple)):
            raise TypeError(
                f"Tuning must map knob names to lists of choices; got {name!r}: "
                f"{choices!r}"
            )
        if not name or not choices:
            raise ValueError(
                f"Tuning knobs need a nonempty name and choices; got {name!r}"
            )
        try:
            snapshot[name] = tuple(json.loads(json.dumps(choices, allow_nan=False)))
        except (TypeError, ValueError) as error:
            raise type(error)(
                f"Tuning choices for {name!r} must be finite JSON values: {error}"
            ) from None
    return snapshot


class _ReadOnlyDict(dict[str, Any]):
    """A dict that refuses changes, so a frozen policy's tuning stays validated.

    Unlike ``types.MappingProxyType``, it still copies, pickles and serializes
    to JSON as a dict does.
    """

    def _read_only(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("SearchPolicy tuning is read-only; create a new SearchPolicy")

    __setitem__ = __delitem__ = clear = popitem = _read_only
    # mypy cannot match one signature to these overloaded methods.
    __ior__ = pop = setdefault = update = _read_only  # type: ignore[assignment]

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (dict(self),)


def _frozen(value: Any) -> Any:
    """Return JSON data with arrays as tuples and objects as read-only dicts."""
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, dict):
        return _ReadOnlyDict({key: _frozen(item) for key, item in value.items()})
    return value


@dataclass(frozen=True, kw_only=True)
class SearchPolicy:
    """Search budget and launch tuning for a :class:`KernelGenerator`.

    Args:
        batch_size: Proposals the author may return per round.
        max_candidates: Search evaluations, counting tuning variants, resumed
            candidates and diagnostic runs, but not final validation or cache
            checks. ``None`` budgets ``max_attempts`` rounds of ``batch_size``
            proposals together with their tuning variants.
        max_no_improvement: Stop after this many consecutive author rounds
            without a better correct candidate; ``None`` does not stop early.
        max_variants: Tuning variants evaluated per proposal, at most 128.
        finalists: Best candidates eligible for the final validation on fresh
            inputs; if one fails, the next is tried.
        resume: Continue the checkpointed search of the same task, validating
            and measuring its retained candidates again.
        tuning: Allowed values for backend proposal fields, or for named
            constants as ``"constants.NAME"``; for example
            ``{"num_warps": (4, 8)}``. Each proposal is evaluated with up to
            ``max_variants`` combinations of these choices. The choices are
            copied as read-only JSON data, with arrays as tuples. Unlike the
            other settings here, they are part of the kernel cache key: a
            kernel cached under different choices is not reused, even if its
            own values are among them.
        screening: How search evaluations run candidates and tuning variants.
            ``"launch"`` registers each candidate as the plugin an engine
            would embed, which applies the plugin's own compilation and checks,
            then launches the kernel directly through the backend, checks it
            on the same inputs and times those launches, without building a
            TensorRT engine; ``"engine"`` builds a trial engine for every
            evaluation. Either way, finalists, cached kernels and final
            validation run inside a trial engine, and only those engine
            measurements decide the ``"eager"`` and ``"tensorrt"``
            objectives. Under ``"tensorrt"``, screens build the TensorRT
            baseline until one search evaluation has built it, so a region
            TensorRT cannot build stops the search at once. Backends without a
            ``launch`` method always build engines.
        auto_fit: When a search candidate needs more shared memory than a
            TensorRT plugin launch allows, let the evaluator compile smaller
            variants of it (one pipeline stage, then halved ``tl.arange`` tile
            constants), and check and time the first that fits within the same
            evaluation instead of rejecting it. The fitted variant passes every
            check an authored candidate does, and it is the kernel recorded
            and accepted. Backends without a fitting method ignore it.
    """

    batch_size: int = 1
    max_candidates: int | None = None
    max_no_improvement: int | None = None
    max_variants: int = 8
    finalists: int = 1
    resume: bool = False
    tuning: dict[str, tuple[Any, ...]] = field(default_factory=dict, hash=False)
    screening: str = "launch"
    auto_fit: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_variants", "finalists"):
            _check(self, name, int)
        for name in ("max_candidates", "max_no_improvement"):
            _check(self, name, int, optional=True)
        _check(self, "resume", bool)
        _check(self, "auto_fit", bool)
        _check(self, "screening", str)
        if self.screening not in ("launch", "engine"):
            raise ValueError(
                "SearchPolicy screening must be 'launch' or 'engine'; got "
                f"{self.screening!r}"
            )
        # Every variant of a proposal is materialized before evaluation.
        if self.max_variants > 128:
            raise ValueError(
                f"SearchPolicy max_variants must be at most 128; got {self.max_variants}"
            )
        object.__setattr__(self, "tuning", _frozen(_snapshot_tuning(self.tuning)))


@dataclass(frozen=True, kw_only=True)
class KernelGenerator:
    """Generate a region kernel during compilation with a logged-in agent CLI.

    Pass it to :func:`fused_kernel`, creating it outside the traced
    ``forward``. The author proposes kernels; worker processes check each
    against the captured PyTorch region and time it, by default launching the
    kernel directly (see :class:`SearchPolicy` ``screening``). The selected
    kernel is checked again on fresh inputs inside a trial TensorRT engine
    before it is cached and registered. Model requests include the captured
    graph, and generated source runs locally.

    Compilation first checks every generated region for what generation
    requires, so that an invalid region fails before any search; a dry run
    logs those failures as warnings, lists them in its report, and calls no
    agent.

    A search that ends without an accepted kernel raises :class:`RegionError`
    and fails the compilation: nothing else runs the region. Kernels accepted
    for other regions stay cached. To proceed, compile again with
    ``SearchPolicy(resume=True)`` to continue the search, or with a larger
    budget, a lower ``min_speedup`` or ``objective="candidate"``; supply a
    kernel through ``fused_kernel(kernel=...)``; or remove the annotation.

    Args:
        provider: Author CLI, ``"claude"`` or ``"codex"``, or a trusted
            ``module:attribute`` provider adapter.
        backend: Kernel language: ``"triton"``, NVIDIA ``"cutile"``,
            ``"cuda_cpp"`` (CUDA C++ compiled through CUDA Python), NVIDIA
            ``"cutedsl"`` (CuTe DSL), or a trusted ``module:attribute`` backend
            adapter.
        model: Model name for the provider; ``None`` keeps the CLI's default.
        instructions: Guidance added to the packaged backend skill.
        instruction_files: Guidance files. They must exist when the generator
            is created and are read again when compilation checks the region,
            before any search, and when each search starts.
        max_attempts: Author rounds per search.
        timeout_s: Time budget for generating one region, in seconds, covering
            authoring, evaluation and final validation. Each generated region
            has its own budget. The default leaves room for several full author
            turns (``author_timeout_s``) and the evaluation of what they
            propose. When the remaining budget shortens a turn and that turn
            runs out of time, the search stops unless another full turn still
            fits. Waiting for another process that is generating the same
            kernel is bounded separately by the same value.
        candidate_timeout_s: Time limit for evaluating one candidate. It
            covers computing the expected outputs of every validation case
            (eager PyTorch, its TF32 re-run under
            ``fp32_matmul_precision="high"``, and the float64 reference), then
            compiling, checking and timing the candidate; large regions take
            minutes. Each author round keeps up to two such limits, and final
            validation one per finalist, out of ``timeout_s``.
        author_timeout_s: Time limit for one author turn. The first request
            of a round, each evidence read or diagnosis, and each protocol
            retry is a turn with its own limit; the round still stops early
            enough to evaluate its proposals.
        max_context_requests: Evidence reads and diagnostic requests allowed
            per author round. A request beyond them breaks the response
            protocol.
        max_protocol_retries: Author turns per round that may repair a reply
            breaking the response protocol, such as a proposal with no
            candidates or a reply the schema rejects, before the round fails.
            The retry names the defect and costs no author round.
        cache_dir: Directory for prompts, captured reference graphs, generated
            source, feedback, logs, checkpoints and accepted kernels. Defaults
            to ``$XDG_CACHE_HOME/torch_tensorrt/region_kernels``, or
            ``~/.cache/torch_tensorrt/region_kernels``. When the provider
            reports what an author turn cost, as the ``claude`` CLI does, the
            run's ``search.json`` totals it, and the total is logged.
        use_cache: Reuse an accepted kernel after validating it again, and
            publish newly accepted kernels. ``False`` neither reads nor
            publishes accepted kernels, but still writes run artifacts.
        num_tests: Random input sets per correctness check.
        benchmark_iters: Kernel launches per timing sample.
        check_nonfinite: Also check inputs containing NaN and infinities: cases
            that inject a NaN or an infinity into each input. On them, an
            output that the injected value reaches (one that eager PyTorch
            makes NaN when the injected element is NaN) may be NaN or infinite
            even where eager PyTorch's is finite, as when ReLU computed as
            ``x * (x > 0)`` turns -inf into NaN instead of 0; a finite value
            there must still be accurate, and outputs that the injected value
            does not reach are checked as on other cases. On every case, any
            value passes wherever eager PyTorch's output is NaN or infinite.
        benchmark_repeats: Interleaved timing samples of each compared
            implementation.
        objective: ``"candidate"`` accepts the fastest correct candidate, even
            one slower than the captured PyTorch region, and then logs a
            warning. ``"eager"`` and ``"tensorrt"`` also require
            ``min_speedup`` over the captured PyTorch region or over a fully
            TensorRT-compiled version of it.
        min_speedup: Required baseline/candidate latency ratio after a
            median-absolute-deviation noise allowance. Values above 1 require
            the ``"eager"`` or ``"tensorrt"`` objective.
        rtol: Relative output tolerance, set together with ``atol``; outputs
            are then compared with eager PyTorch's at these tolerances. When
            both are ``None``, outputs are compared with a float64 evaluation
            of the reference and may deviate by a small multiple of the error
            of the PyTorch reference itself. If the reference cannot be
            evaluated without rounding, validation fails rather than compare
            with eager PyTorch.
        atol: Absolute output tolerance; see ``rtol``.
        fp32_matmul_precision: How exact float32 matrix multiplication and
            convolution must be, as in ``torch.set_float32_matmul_precision``.
            ``"highest"`` holds candidates to the error of the PyTorch
            reference in IEEE float32, with TF32 off for cuDNN convolutions
            too, although PyTorch's defaults let them use it. ``"high"`` also
            allows the error that the reference has with TF32 matmuls and
            convolutions, which TensorRT engines use unless
            ``disable_tf32=True``. Has no effect with ``rtol`` and ``atol``.
            Either way, eager PyTorch is timed at PyTorch's default TF32
            settings.
        search: Search budget and launch tuning. Its ``tuning`` choices are
            part of the kernel cache key; its other settings are not.
        diagnostics: Include compiler and execution facts in feedback, and let
            the author request a diagnostic run of the best kernel so far.
        aligned_io: Assume every plugin input and output pointer is 16-byte
            aligned, as TensorRT allocations are, so ahead-of-time builds can
            use vector loads and stores. Unsafe if the engine hands the plugin
            a sub-buffer view at an odd offset, as TensorRT can when it fuses a
            concatenation into its inputs' buffers. It changes the cache key.
            Without it, evaluations launch kernels directly on pointers aligned
            only to their element size, so a kernel that assumes more fails.
        allow_generation: Search for a new kernel when no cached kernel passes
            revalidation. ``False`` only reuses a cached kernel and otherwise
            fails compilation with a :class:`RegionError` before any provider
            call, as CI or production builds may require; setting the
            environment variable ``TORCH_TENSORRT_REGION_GENERATION=0`` does
            the same for every generator. Requires ``use_cache=True``.
        validation_inputs: Real model inputs to validate candidates on: a
            reusable iterable of batches, such as a list or a
            ``torch.utils.data.DataLoader``. Each batch is a tensor or a tuple
            of tensors for the positional inputs given to
            ``torch_tensorrt.compile``, followed by a dict of the keyword
            inputs if it was given ``kwarg_inputs``; a dict alone holds only
            keyword inputs. Compilation runs the model on the first 8 batches,
            or fewer once the region inputs kept from them reach 1 GiB, keeps
            the tensors that reach the region, and checks every candidate on
            them as well as on synthetic inputs derived from them. Batches
            must have the compiled shapes and dtypes and are moved to the
            inputs' device.
            The kept tensors are saved with the run but are not part of the
            cache key: a cached kernel is reused once it passes a check on
            them, so new weights or other batches need no new search. Failure
            feedback can show the author values from them. ``None`` uses the
            example inputs given to ``torch_tensorrt.compile`` as one batch
            when they are tensors rather than ``Input`` shape specifications
            and the model does not write to its buffers or inputs; if they
            cannot run the model, a warning is logged and candidates are
            validated on synthetic inputs alone. A region whose tensors from
            them would bring those kept past 1 GiB keeps none: a warning is
            logged and it is validated as a region without them.
    """

    provider: str = "claude"
    backend: str = "triton"
    model: str | None = None
    instructions: str | None = None
    instruction_files: tuple[str | Path, ...] = ()
    max_attempts: int = 4
    timeout_s: float = 1800
    candidate_timeout_s: float = 300
    author_timeout_s: float = 300
    max_context_requests: int = 4
    max_protocol_retries: int = 1
    cache_dir: str | Path | None = None
    use_cache: bool = True
    num_tests: int = 3
    benchmark_iters: int = 20
    check_nonfinite: bool = True
    benchmark_repeats: int = 5
    objective: str = "candidate"
    min_speedup: float = 1.0
    rtol: float | None = None
    atol: float | None = None
    fp32_matmul_precision: str = "highest"
    search: SearchPolicy = field(default_factory=SearchPolicy)
    diagnostics: bool = False
    aligned_io: bool = False
    allow_generation: bool = True
    # Batches are data, not configuration: equality, hashing and the cache key
    # ignore them.
    validation_inputs: Iterable[Any] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        for name in ("provider", "backend", "objective"):
            _check(self, name, str)
        for name in ("model", "instructions"):
            _check(self, name, str, optional=True)
        for name in (
            "max_attempts",
            "num_tests",
            "benchmark_iters",
            "benchmark_repeats",
        ):
            _check(self, name, int)
        _check(self, "max_context_requests", int, positive=False)
        _check(self, "max_protocol_retries", int, positive=False)
        for name in ("timeout_s", "candidate_timeout_s", "author_timeout_s"):
            _check(self, name, float)
        _check(self, "min_speedup", float)
        for name in ("rtol", "atol"):
            _check(self, name, float, positive=False, optional=True)
        for name in ("use_cache", "check_nonfinite", "diagnostics", "allow_generation"):
            _check(self, name, bool)
        _check(self, "aligned_io", bool)
        if not self.allow_generation and not self.use_cache:
            raise ValueError(
                "KernelGenerator allow_generation=False requires use_cache=True; "
                "without generation, only a cached kernel can be used"
            )
        if not isinstance(self.search, SearchPolicy):
            raise TypeError(
                f"KernelGenerator search must be a SearchPolicy; got {self.search!r}"
            )
        if isinstance(self.instruction_files, (str, bytes, Path)):
            raise TypeError(
                "KernelGenerator instruction_files must be a sequence of paths; "
                f"got {self.instruction_files!r}"
            )
        batches = self.validation_inputs
        # Every compilation iterates the batches again, which a one-shot
        # iterator cannot do; a bare tensor would be iterated row by row.
        if batches is not None and (
            not isinstance(batches, Iterable)
            or isinstance(batches, (Iterator, torch.Tensor, str, bytes, Mapping))
        ):
            raise TypeError(
                "KernelGenerator validation_inputs must be a reusable iterable of "
                "batches, such as a list or a DataLoader; got "
                f"{type(batches).__name__}"
            )

        for name, builtins in (("provider", _PROVIDERS), ("backend", _BACKENDS)):
            value = getattr(self, name)
            if value not in builtins and not IMPORT_PATH.fullmatch(value):
                raise ValueError(
                    f"KernelGenerator {name} must be one of {builtins} or a "
                    f"'module:attribute' import path; got {value!r}"
                )
        if self.objective not in ("candidate", "eager", "tensorrt"):
            raise ValueError(
                "KernelGenerator objective must be 'candidate', 'eager' or "
                f"'tensorrt'; got {self.objective!r}"
            )
        if self.model is not None and not self.model.strip():
            raise ValueError("KernelGenerator model must not be blank")
        if self.min_speedup < 1:
            raise ValueError(
                f"KernelGenerator min_speedup must be at least 1; got {self.min_speedup!r}"
            )
        if self.min_speedup != 1 and self.objective == "candidate":
            raise ValueError(
                "KernelGenerator min_speedup requires objective='eager' or "
                "'tensorrt'; the 'candidate' objective has no baseline"
            )
        if (self.rtol is None) != (self.atol is None):
            raise ValueError("KernelGenerator rtol and atol must be set together")
        _check(self, "fp32_matmul_precision", str)
        if self.fp32_matmul_precision not in ("highest", "high"):
            raise ValueError(
                "KernelGenerator fp32_matmul_precision must be 'highest' or 'high'; "
                f"got {self.fp32_matmul_precision!r}"
            )
        files = tuple(Path(p).expanduser().resolve() for p in self.instruction_files)
        for path in files:
            if not path.is_file():
                raise ValueError(
                    f"KernelGenerator instruction file {path} is not an existing file"
                )
        object.__setattr__(self, "instruction_files", files)
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).expanduser())
        if self.backend in _BACKENDS:
            # Extension backends are imported only when generation runs, so
            # only built-in tuning schemas can be checked this early.
            from torch_tensorrt.dynamo.region_compilation._generation._backends import (
                validate_tuning_space,
            )

            validate_tuning_space(self.backend, self.search.tuning)
