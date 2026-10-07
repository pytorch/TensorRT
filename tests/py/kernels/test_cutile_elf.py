# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""cuTile PTX extraction must distinguish file payloads from reserved memory."""

import struct

import pytest
import torch

import torch_tensorrt.kernels as ttk
from torch_tensorrt.kernels._cutile import extract_ptx_from_cubin

from .conftest import (
    assert_ran_in_engine,
    compile_op,
    register_once,
    skip_no_cuda,
    skip_no_cutile,
    skip_no_qdp,
)

_PTX = b".version 9.0\n.target sm_120\n.address_size 64\n.visible .entry kernel() { ret; }\n"


def _cubin():
    names = b"\x00.shstrtab\x00.nv_debug_ptx_txt\x00.nv.shared.kernel\x00"
    blob = bytearray(64)
    blob[:7] = b"\x7fELF\x02\x01\x01"
    names_offset = len(blob)
    blob.extend(names)
    ptx_offset = len(blob)
    blob.extend(_PTX)
    table_offset = len(blob)
    struct.pack_into("<Q", blob, 40, table_offset)
    struct.pack_into("<HHH", blob, 58, 64, 4, 1)

    sections = [
        (0, 0, 0, 0),
        (names.index(b".shstrtab"), 3, names_offset, len(names)),
        (names.index(b".nv_debug_ptx_txt"), 1, ptx_offset, len(_PTX)),
        # SHT_NOBITS occupies no file bytes even when its memory size is large.
        (names.index(b".nv.shared.kernel"), 8, table_offset, 1024 * 1024),
    ]
    for name, section_type, offset, size in sections:
        blob.extend(
            struct.pack(
                "<IIQQQQIIQQ", name, section_type, 0, 0, offset, size, 0, 0, 1, 0
            )
        )
    return blob, table_offset


def test_large_nobits_section_does_not_hide_embedded_ptx():
    cubin, _ = _cubin()

    assert extract_ptx_from_cubin(bytes(cubin)) == _PTX.decode()


@pytest.mark.parametrize("section", [1, 2], ids=["string_table", "ptx"])
def test_string_table_and_ptx_must_have_file_payloads(section):
    cubin, table_offset = _cubin()
    struct.pack_into("<I", cubin, table_offset + section * 64 + 4, 8)

    assert extract_ptx_from_cubin(bytes(cubin)) is None


@pytest.mark.parametrize("section", [1, 2], ids=["string_table", "ptx"])
@pytest.mark.parametrize("field", [24, 32], ids=["offset", "size"])
def test_file_backed_section_bounds_are_still_checked(section, field):
    cubin, table_offset = _cubin()
    struct.pack_into("<Q", cubin, table_offset + section * 64 + field, len(cubin) + 1)

    assert extract_ptx_from_cubin(bytes(cubin)) is None


try:
    import cuda.tile as ct

    @ct.kernel
    def _shared_memory_matmul(a, b, out):
        x = ct.load(a, index=(0, 0), shape=(64, 64))
        y = ct.load(b, index=(0, 0), shape=(64, 64))
        ct.store(out, index=(0, 0), tile=ct.matmul(x, y))

except ImportError:
    ct = None


@skip_no_cuda
@skip_no_cutile
@skip_no_qdp
def test_shared_memory_matmul_compiles_and_runs_in_engine():
    op_name = "ttk_test::cutile_shared_memory_matmul"

    def _meta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)

    register_once(
        op_name,
        lambda: ttk.cutile_op(
            op_name,
            _shared_memory_matmul,
            {"a": "fp16", "b": "fp16", "out": "fp16"},
            _meta,
            ndim=2,
            grid=lambda inputs, outputs: 1,
            supports_dynamic_shapes=False,
        ),
    )
    a, b = [torch.randn(64, 64, device="cuda", dtype=torch.float16) for _ in range(2)]
    compiled = compile_op(op_name, [a, b])
    assert_ran_in_engine(compiled, op_name)

    with torch.no_grad():
        torch.testing.assert_close(compiled(a, b), a @ b, atol=1e-2, rtol=1e-2)
