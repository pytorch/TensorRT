# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Workloads for the Auto-Tuner tests.

``create_fake`` builds a CPU-only workload whose latency, accuracy and failures
are scripted per knob value, so session behavior is deterministic.
``create_mlp`` is a small real module for GPU tests.
"""

import dataclasses
import hashlib
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_tensorrt.autotune import (
    Candidate,
    EditableFile,
    Knob,
    ModuleWorkload,
    Timing,
    Workload,
    load_module,
)
from torch_tensorrt.region import (
    GeneratedKernel,
    MissingKernel,
    RegionError,
    record_generated_kernel,
    record_missing_kernel,
)

FAKE_KNOBS = (
    Knob(
        "runtime.cudagraphs",
        (False, True),
        False,
        "replay a graph",
        ("launch_overhead",),
    ),
    Knob("engine.precision", ("fp32", "fp16"), "fp32", "precision", ("precision",)),
    Knob("engine.optimization_level", (3, 5), 3, "builder level", ("dominant_kernel",)),
    Knob("workload.variant", ("a", "b", "c"), "a", "workload option"),
)


def kernel_knob(name: str) -> Knob:
    """A knob whose prepared value ``generated`` gives the region ``name`` a kernel."""
    return Knob(
        f"kernels.{name}",
        ("native", "generated"),
        "native",
        f"generate a kernel for the region {name!r}",
        ("dominant_kernel",),
        prepare=("generated",),
    )


KERNEL_KNOB = kernel_knob("fused")
# What the generator's environment switch is set to when generation is off.
GENERATION_SWITCH = "TORCH_TENSORRT_REGION_GENERATION"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FakeKernelCache:
    """A kernel cache in a JSON file, mapping generation tasks to kernel digests.

    It stands in for a KernelGenerator's cache directory: a build with
    generation on reuses a cached kernel or caches a new one, and a build
    with generation off fails on a task that has no cached kernel.
    """

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None

    def load(self) -> dict[str, str]:
        if self.path is None or not self.path.is_file():
            return {}
        return json.loads(self.path.read_text())

    def store(self, task: str, kernel: str) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({**self.load(), task: kernel}))


# How much faster than its eager baseline every fake region kernel runs.
KERNEL_SPEEDUP = 2.0


def record_fake_kernels(
    values: Mapping[str, Any],
    kernels: Mapping[str, Mapping[str, Any]],
    cache: FakeKernelCache,
    regenerates: set,
) -> None:
    """Register the kernel of every region whose knob is ``generated``.

    ``kernels`` maps a knob's region name to its settings: ``region`` (the
    name the region has in the model, by default its knob's), ``task`` (the
    name of its computation, by default its knob's; regions with the same
    computation share their generation tasks). Every kernel runs its region
    :data:`KERNEL_SPEEDUP` times faster than its eager baseline. A region's
    generation task, and so its kernel, depends on the optimization level, as
    engine build options are part of a real task. With a cache, a build with
    generation off loads cached kernels and records a missing kernel (then
    fails) for a task without one; a build with generation on reuses a cached
    kernel unless its option is in ``regenerates``. Without a cache a kernel
    is "cached" when generation is off and "new" otherwise.
    """
    switched_off = os.environ.get(GENERATION_SWITCH) == "0"
    level = values["engine.optimization_level"]
    for index, (name, settings) in enumerate(kernels.items()):
        if values.get(f"kernels.{name}") != "generated":
            continue
        region = settings.get("region", name)
        computation = settings.get("task", name)
        task = _sha(f"{computation} task at level {level}")
        cached = cache.load().get(task)
        option = f"kernels.{name}=generated"
        if cache.path is None:
            origin = "cached" if switched_off else "new"
            kernel = _sha(f"{computation} kernel at level {level}")
        elif switched_off and cached is None:
            record_missing_kernel(
                MissingKernel(
                    region=region,
                    region_id=f"{region}#{index}",
                    task_sha256=task,
                    reason="no accepted kernel is cached for this task",
                )
            )
            raise RegionError(
                f"Kernel generation for region '{region}#{index}' is disabled by "
                f"{GENERATION_SWITCH}=0, and no accepted kernel is cached for this "
                "task"
            )
        elif cached is not None and (switched_off or option not in regenerates):
            origin, kernel = "cached", cached
        else:
            origin = "new"
            kernel = _sha(f"{computation} kernel at level {level} for {option}")
            cache.store(task, kernel)
        record_generated_kernel(
            GeneratedKernel(
                region=region,
                region_id=f"{region}#{index}",
                task_sha256=task,
                kernel_sha256=kernel,
                backend="triton",
                provider="fake",
                origin=origin,
                generation_s=1.5,
                latency_ms=0.1,
                baseline="eager",
                baseline_latency_ms=0.1 * KERNEL_SPEEDUP,
                speedup=KERNEL_SPEEDUP,
            )
        )


# Options (comma-separated "knob=value") whose build crashes the worker. An
# environment variable, unlike the workload's arguments, may differ between
# runs of one session.
CRASH_ENV = "AUTOTUNE_TEST_CRASH"


def crash() -> None:
    """Die of SIGSEGV, as a native library would, without leaving a core file."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), signal.SIGSEGV)


def idle() -> None:
    while True:
        time.sleep(0.05)


def start_idle_threads(count: int) -> None:
    """Start background threads, like a progress bar's monitor or a thread pool."""
    for _ in range(count):
        threading.Thread(target=idle, daemon=True).start()


class FakeCandidate(Candidate):
    def __init__(
        self,
        milliseconds: float,
        output: torch.Tensor,
        wobble_calls: int = 0,
        export_error: Exception | None = None,
    ) -> None:
        self.milliseconds = milliseconds
        self.output = output
        self.units = 2
        # The first wobble_calls calls alternate between 1x and 3x the time,
        # like a run that shares the GPU with another job.
        self.wobble_calls = wobble_calls
        self.calls = 0
        self.export_error = export_error

    def __call__(self) -> torch.Tensor:
        factor = 3.0 if self.calls < self.wobble_calls and self.calls % 2 else 1.0
        self.calls += 1
        time.sleep(self.milliseconds * factor / 1000)
        return self.output.clone()

    def describe(self) -> dict[str, Any]:
        return {"trt_engines": 1, "torch_partitions": 0, "plugin_regions": 0}

    def export(self, directory):
        if self.export_error is not None:
            raise self.export_error
        (directory / "fake.txt").write_text(f"{self.milliseconds}\n")
        return ["fake.txt"]


class FakeWorkload(Workload):
    """A CPU workload whose candidates sleep; every behavior is scripted by option.

    An option is ``"knob=value"``, for example ``"engine.precision=fp16"``.

    * ``milliseconds``: how long one call (two work units) takes;
      ``speedups`` maps options to factors of that time.
    * ``broken``: options whose build fails; ``inaccurate``: options that
      change the output.
    * ``baseline_changes``: the workload's own baseline config;
      ``reference_fails``: computing the reference outputs fails.
    * ``crashes``: options whose build kills the worker (SIGSEGV), as do
      those in the ``AUTOTUNE_TEST_CRASH`` environment variable; ``hangs``:
      options whose build never returns; ``threads`` background threads start
      before such a build.
    * ``unexportable`` and ``export_breaks``: options whose export raises
      NotImplementedError or RuntimeError.
    * ``requires`` maps a knob's name to its ``requires``.
    * ``flops_per_unit`` and ``peak_tflops`` give candidates a compute floor.
    * ``wobble_calls``: the first that many calls alternate between 1x and
      3x the time, like a run sharing the GPU.
    * ``editable_file``: declare the editable file ``speed``
      (``_fake_editable.py``), whose ``SPEED`` scales the latency.
    * ``kernels`` maps region names to settings (see
      :func:`record_fake_kernels`): each declares ``kernels.<name>``, whose
      prepared value ``generated`` registers the region's kernel.
      ``kernel_knob`` declares
      ``kernels.fused`` (``KERNEL_KNOB``) alone. ``kernel_cache`` is the JSON
      file of a fake kernel cache, so that a build with generation off fails
      on a missing kernel; builds of ``regenerates`` options replace cached
      kernels.
    """

    name = "fake"

    def __init__(
        self,
        milliseconds: float = 6.0,
        speedups: Mapping[str, float] | None = None,
        broken: tuple = (),
        inaccurate: tuple = (),
        baseline_changes: Mapping[str, Any] | None = None,
        reference_fails: bool = False,
        wobble_calls: int = 0,
        editable_file: bool = False,
        crashes: tuple = (),
        hangs: tuple = (),
        unexportable: tuple = (),
        export_breaks: tuple = (),
        requires: Mapping[str, Any] | None = None,
        flops_per_unit: float | None = None,
        peak_tflops: float | None = None,
        threads: int = 0,
        kernel_knob: bool = False,
        kernels: Mapping[str, Mapping[str, Any]] | None = None,
        kernel_cache: str | None = None,
        regenerates: tuple = (),
    ) -> None:
        self.kernels = dict(kernels or ({"fused": {}} if kernel_knob else {}))
        self.kernel_cache = FakeKernelCache(kernel_cache)
        self.regenerates = set(regenerates)
        self.milliseconds = milliseconds
        self.speedups = dict(speedups or {})
        self.broken = set(broken)
        self.inaccurate = set(inaccurate)
        self.baseline_changes = dict(baseline_changes or {})
        self.reference_fails = reference_fails
        self.wobble_calls = wobble_calls
        self.editable_file = editable_file
        self.crashes = set(crashes)
        self.hangs = set(hangs)
        self.unexportable = set(unexportable)
        self.export_breaks = set(export_breaks)
        self.requires = dict(requires or {})
        self.flops_per_unit = flops_per_unit
        self.peak_tflops = peak_tflops
        self.threads = threads

    def knobs(self):
        return tuple(
            (
                dataclasses.replace(knob, requires=self.requires[knob.name])
                if knob.name in self.requires
                else knob
            )
            for knob in FAKE_KNOBS + tuple(kernel_knob(name) for name in self.kernels)
        )

    def baseline(self):
        return dict(self.baseline_changes)

    def timing(self):
        return Timing(warmup=1, rounds=5, calls=2, profile_calls=0)

    def accuracy_defaults(self):
        return {"max_abs_error": ("<=", 1e-3)}

    def editable_files(self):
        if not self.editable_file:
            return ()
        path = Path(__file__).with_name("_fake_editable.py")
        return (EditableFile("speed", str(path), "SPEED scales the latency"),)

    def reference(self):
        if self.reference_fails:
            raise RuntimeError("no reference today")
        return torch.ones(4)

    def build(self, values: Mapping[str, Any]) -> Candidate:
        options = {f"{name}={value}" for name, value in values.items()}
        crashes = self.crashes | set(
            filter(None, os.environ.get(CRASH_ENV, "").split(","))
        )
        if options & (crashes | self.hangs):
            start_idle_threads(self.threads)
        if options & crashes:
            crash()
        if options & self.hangs:
            time.sleep(3600)
        if options & self.broken:
            raise RuntimeError(f"cannot build {sorted(options & self.broken)}")
        record_fake_kernels(
            values,
            self.kernels,
            self.kernel_cache,
            self.regenerates,
        )
        milliseconds = self.milliseconds
        for option, factor in self.speedups.items():
            if option in options:
                milliseconds *= factor
        if values.get("edit.speed"):
            milliseconds *= load_module(values["edit.speed"], "fake_speed").SPEED
        output = torch.ones(4) * (1.5 if options & self.inaccurate else 1.0)
        export_error = None
        if options & self.unexportable:
            export_error = NotImplementedError("this build cannot be saved")
        elif options & self.export_breaks:
            export_error = RuntimeError("the disk is full")
        candidate = FakeCandidate(milliseconds, output, self.wobble_calls, export_error)
        candidate.flops_per_unit = self.flops_per_unit
        candidate.peak_tflops = self.peak_tflops
        return candidate


def create_fake(**kwargs: Any) -> FakeWorkload:
    return FakeWorkload(**kwargs)


def create_mlp(width: int = 256, depth: int = 4, batch: int = 8) -> ModuleWorkload:
    torch.manual_seed(0)
    layers: list[torch.nn.Module] = []
    for _ in range(depth):
        layers += [torch.nn.Linear(width, width), torch.nn.GELU()]
    model = torch.nn.Sequential(*layers).cuda()
    inputs = [torch.randn(batch, width, device="cuda")]
    return ModuleWorkload(
        model,
        inputs,
        name="mlp",
        compile_options={"min_block_size": 1},
        timing=Timing(warmup=2, rounds=5, calls=4, profile_calls=2),
    )
