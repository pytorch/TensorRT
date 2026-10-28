# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""A TensorRT build that fails says why, instead of raising a bare AssertionError."""

from unittest import mock

import pytest
import tensorrt as trt
import torch
from torch_tensorrt.logging import TRT_LOGGER


def test_capture_errors_collects_only_errors_reported_inside_it():
    TRT_LOGGER.log(trt.ILogger.Severity.ERROR, "before")
    with TRT_LOGGER.capture_errors() as outer:
        TRT_LOGGER.log(trt.ILogger.Severity.WARNING, "a warning")
        with TRT_LOGGER.capture_errors() as inner:
            TRT_LOGGER.log(trt.ILogger.Severity.ERROR, "inside")
        TRT_LOGGER.log(trt.ILogger.Severity.ERROR, "outer only")
    TRT_LOGGER.log(trt.ILogger.Severity.ERROR, "after")
    assert outer == ["inside", "outer only"] and inner == ["inside"]
    # Two captures that hold equal lists when the first one ends.
    with TRT_LOGGER.capture_errors() as first:
        with TRT_LOGGER.capture_errors() as second:
            pass
        TRT_LOGGER.log(trt.ILogger.Severity.ERROR, "first only")
    assert first == ["first only"] and second == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="builds a TensorRT network")
@pytest.mark.parametrize("version_compatible", [False, True])
def test_a_refused_build_raises_tensorrt_s_own_reason(version_compatible):
    import torch_tensorrt

    def refuse(builder, network, config):
        TRT_LOGGER.log(
            trt.ILogger.Severity.ERROR,
            "Networks with FP8 Q/DQ layers require hardware with FP8 support.",
        )
        return None

    model = torch.nn.Linear(8, 8).eval().cuda()
    inputs = (torch.randn(2, 8, device="cuda"),)
    program = torch.export.export(model, inputs)
    # Both build calls: version-compatible and TensorRT-RTX builds use the second, and an RTX
    # Builder may not have the first.
    with (
        mock.patch.object(trt.Builder, "build_engine_with_config", refuse, create=True),
        mock.patch.object(trt.Builder, "build_serialized_network", refuse),
    ):
        with pytest.raises(RuntimeError, match="require hardware with FP8 support"):
            torch_tensorrt.dynamo.compile(
                program,
                arg_inputs=list(inputs),
                min_block_size=1,
                version_compatible=version_compatible,
                cache_built_engines=False,
                reuse_cached_engines=False,
            )
