import unittest

import pytest
import torch
import torch.nn as nn
import torch_tensorrt as torchtrt
from parameterized import parameterized
from torch.export import Dim
from torch_tensorrt import Input
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

assertions = unittest.TestCase()


@pytest.mark.unit
@parameterized.expand(
    [
        ((5,), (5,)),
        (
            (
                2,
                3,
            ),
            (
                2,
                3,
            ),
        ),
    ]
)
def test_atan2_out_static_shape(input_shape, out_shape):
    class atan2(torch.nn.Module):
        def forward(self, lhs_val, rhs_val, out):
            return torch.ops.aten.atan2.out(lhs_val, rhs_val, out=out)

    model = atan2().eval().cuda()
    inputs = (
        torch.randn(input_shape).cuda(),
        torch.randn(input_shape).cuda(),
        torch.randn(out_shape).cuda(),
    )
    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
    }

    trt_model = torchtrt.compile(model, **compile_spec)
    py_outputs = model(*inputs)
    trt_outputs = trt_model(*inputs)
    cos_sim = cosine_similarity(py_outputs, trt_outputs)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_atan2_out_static_shape model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@pytest.mark.unit
@parameterized.expand(
    [
        (
            (
                1,
                2,
            ),
            (2, 3),
            (2, 4),
        ),
    ]
)
def test_atan2_out_dynamic_shape(min_shape, opt_shape, max_shape):
    class atan2(torch.nn.Module):
        def forward(self, lhs_val, rhs_val, out):
            return torch.ops.aten.atan2.out(lhs_val, rhs_val, out=out)

    model = atan2().eval().cuda()
    input_spec = [
        Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
        ),
        Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
        ),
        Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
        ),
    ]

    compile_spec = {
        "inputs": input_spec,
        "ir": "dynamo",
        "min_block_size": 1,
    }

    trt_model = torchtrt.compile(model, **compile_spec)
    inputs = (
        torch.randn(max_shape).cuda(),
        torch.randn(max_shape).cuda(),
        torch.randn(max_shape).cuda(),
    )
    py_outputs = model(*inputs)
    trt_outputs = trt_model(*inputs)
    cos_sim = cosine_similarity(py_outputs, trt_outputs)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_atan2_out_dynamic_shape model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@parameterized.expand(
    [
        ((32, 8, 128, 64), (32, 8, 128, 64), True, None),
    ]
)
def test_sdpa_static_shape(query_shape, key_shape, is_causal, scale):
    class SDPA(nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, None, 0.0, is_causal=is_causal, scale=scale
            )

    model = SDPA().eval().cuda()

    query = torch.randn(query_shape, dtype=torch.float16).cuda()
    key = torch.randn(key_shape, dtype=torch.float16).cuda()
    value = torch.randn(key_shape, dtype=torch.float16).cuda()
    inputs = (query, key, value)
    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
    }

    trt_model = torchtrt.compile(model, **compile_spec)
    py_outputs = model(*inputs)
    trt_outputs = trt_model(*inputs)
    cos_sim = cosine_similarity(py_outputs, trt_outputs)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_sdpa_static_shape model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )


@parameterized.expand(
    [
        (True, None),
        (True, 0.1),
        (False, None),
    ]
)
def test_sdpa_dynamic_shape(is_causal, scale):
    class SDPA(nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, None, 0.0, is_causal=is_causal, scale=scale
            )

    model = SDPA().eval().cuda()

    # N: batch_size
    dyn_N = Dim("dyn_N", min=2, max=4)

    # query tensor shape (N, ..., Hq, L, E)
    query = torch.randn((3, 3, 4, 64), dtype=torch.float16).cuda()
    # key tensor shape (N,...,H, S, E)
    key = torch.randn((3, 3, 4, 64), dtype=torch.float16).cuda()
    # value tensor shape (N, ..., H, S, Ev)
    value = torch.randn((3, 3, 4, 64), dtype=torch.float16).cuda()

    dynamic_shapes = {"query": {0: dyn_N}, "key": {0: dyn_N}, "value": {0: dyn_N}}
    inputs = (query, key, value)

    exp_program = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)

    compile_spec = {
        "inputs": inputs,
        "ir": "dynamo",
        "min_block_size": 1,
    }
    trt_model = torchtrt.dynamo.compile(exp_program, **compile_spec)
    py_outputs = model(*inputs)
    trt_outputs = trt_model(*inputs)
    cos_sim = cosine_similarity(py_outputs, trt_outputs)
    assertions.assertTrue(
        cos_sim > COSINE_THRESHOLD,
        msg=f"test_sdpa_dynamic_shape model TRT outputs don't match with the pytorch model. Cosine sim score: {cos_sim} Threshold: {COSINE_THRESHOLD}",
    )
