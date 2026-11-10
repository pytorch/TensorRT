# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental: the Auto-Tuner, which tunes a Torch-TensorRT workload by measurement.

You provide a workload (a model, representative inputs and a definition of
correct output), a goal (lower latency or higher throughput), limits
(accuracy and memory) and a budget. A :class:`TuningSession` then:

1. measures the baseline config in a fresh worker process, and measures it
   a few more times to learn how much results vary between runs (the noise);
   then builds each knob value that needs preparing once, such as a value
   that generates region kernels, and pins every later run that uses it to
   the kernels that build generated;
2. asks a planner for the next experiment: changes to the best config so
   far, chosen among the workload's knobs in five layers (graph and regions,
   TensorRT engine, kernel generation, runtime execution and the workload's
   own options) and, if the workload allows it, edits of fenced source files;
3. checks the proposal, then builds, checks, times and profiles the
   candidate in a fresh worker process, and accepts it only if it meets the
   accuracy and memory limits and improves the goal's metric by more than
   the noise: first in its own run, then head-to-head against the best
   config;
4. repeats until the goal, the budget or the planner ends the search;
5. runs a final check of the best config against the baseline, and exports
   the winner.

You get the winner's validated artifact, a before/after report and a
reproducible config; if nothing beats the baseline, the baseline is the
winner. Running the artifact needs no planner or LLM.
"""

from torch_tensorrt.autotune._code_edits import EditableFile, load_module
from torch_tensorrt.autotune._knobs import STANDARD_KNOBS, Knob
from torch_tensorrt.autotune._planner import (
    LLMPlanner,
    Planner,
    Proposal,
    RuleBasedPlanner,
)
from torch_tensorrt.autotune._session import (
    TuningError,
    TuningResult,
    TuningSession,
)
from torch_tensorrt.autotune._settings import (
    Budget,
    Goal,
    Limits,
    NoiseControl,
    Timing,
    WorkloadFactory,
)
from torch_tensorrt.autotune._workload import (
    Candidate,
    ModuleWorkload,
    Workload,
    compare_outputs,
    describe_module,
)

__all__ = [
    "Budget",
    "Candidate",
    "EditableFile",
    "Goal",
    "Knob",
    "LLMPlanner",
    "Limits",
    "ModuleWorkload",
    "NoiseControl",
    "Planner",
    "Proposal",
    "RuleBasedPlanner",
    "STANDARD_KNOBS",
    "Timing",
    "TuningError",
    "TuningResult",
    "TuningSession",
    "Workload",
    "WorkloadFactory",
    "compare_outputs",
    "describe_module",
    "load_module",
]
