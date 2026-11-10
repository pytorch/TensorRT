# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""region_generation: the KernelGenerator settings every generated region uses.

Run with ``python -m pytest tools/llm/tests/test_region_generation.py``; no
GPU, model download or agent is needed (a KernelGenerator is only its
settings until a compilation uses it).
"""

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import region_generation  # noqa: E402
from torch_tensorrt.region import KernelGenerator  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
# What every generated region of these workloads asks for.
SHARED = dict(
    provider="claude",
    backend="triton",
    model=None,
    objective="tensorrt",
    min_speedup=1.0,
    timeout_s=1800.0,
    aligned_io=True,
    diagnostics=True,
    use_cache=True,
    allow_generation=True,
)


def assert_shared(generator, cache_dir, notes=None, precision="highest"):
    assert isinstance(generator, KernelGenerator)
    for name, value in SHARED.items():
        assert getattr(generator, name) == value, name
    assert generator.cache_dir == Path(cache_dir)
    expected = () if notes is None else ((HERE / notes).resolve(),)
    assert generator.instruction_files == expected
    assert generator.fp32_matmul_precision == precision


def test_make_generator_has_the_shared_settings(tmp_path):
    assert_shared(region_generation.make_generator(tmp_path), tmp_path)
    notes = "qwen3_decode_kernel_notes.md"
    generator = region_generation.make_generator(str(tmp_path), notes)
    assert_shared(generator, tmp_path, notes)
    # A path names its file as it is; a bare name is a file of tools/llm.
    path = HERE / notes
    assert region_generation.make_generator(tmp_path, path) == generator
    assert region_generation.notes_path(notes) == path
    high = region_generation.make_generator(
        tmp_path, notes, fp32_matmul_precision="high"
    )
    assert_shared(high, tmp_path, notes, "high")
    with pytest.raises(ValueError, match="instruction file"):
        region_generation.make_generator(tmp_path, "no_such_notes.md")
    described = region_generation.describe(high)
    assert described == {
        "kernel_cache": str(tmp_path),
        "generator_notes": [notes],
        "generator_fp32_matmul_precision": "high",
    }


def test_the_qwen3_decode_generator_is_unchanged(tmp_path):
    """The kernels already cached for the Qwen3 decode blocks stay valid.

    The cache key includes the generator's settings and notes: the decode
    blocks' generator must stay exactly the one their cached kernels were
    generated with.
    """
    before = KernelGenerator(
        provider="claude",
        backend="triton",
        model=None,
        instruction_files=(str(HERE / "qwen3_decode_kernel_notes.md"),),
        timeout_s=1800.0,
        cache_dir=str(tmp_path),
        objective="tensorrt",
        min_speedup=1.0,
        aligned_io=True,
        diagnostics=True,
    )
    generator = region_generation.make_generator(
        str(tmp_path), "qwen3_decode_kernel_notes.md"
    )
    assert generator == before
    assert repr(generator) == repr(before)


def test_kernel_cache_is_the_argument_else_the_environment_else_the_default(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(region_generation.KERNEL_CACHE_ENV, raising=False)
    assert region_generation.kernel_cache() == "/dev/shm/exp/kcache"
    assert region_generation.DEFAULT_KERNEL_CACHE == "/dev/shm/exp/kcache"
    monkeypatch.setenv("TORCHTRT_KERNEL_CACHE", str(tmp_path / "env"))
    assert region_generation.kernel_cache() == str(tmp_path / "env")
    assert region_generation.kernel_cache(str(tmp_path / "arg")) == str(
        tmp_path / "arg"
    )


def test_notes_files_hold_measured_facts_and_no_machine_paths():
    import modernbert_kernels
    import nemotron_decode
    import qwen35_gdn

    notes = {
        "qwen3_decode_kernel_notes.md",
        modernbert_kernels.GENERATION_NOTES,
        qwen35_gdn.GENERATION_NOTES,
        nemotron_decode.GENERATION_NOTES,
    }
    assert notes == {path.name for path in HERE.glob("*_kernel_notes.md")}
    for name in sorted(notes):
        text = (HERE / name).read_text()
        assert "Measured" in text or "measured" in text, name
        assert not re.search(r"/(dev/shm|workspace|tmp|root|home)/", text), name
        assert len(text) < 3000, name
